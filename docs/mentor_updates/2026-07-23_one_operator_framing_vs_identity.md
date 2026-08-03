# Result: One context→answer operator — framing bends its coordinates, speaker identity barely moves it

Combined writeup (2026-07-23): merges the 2026-07-21 framing-ablations update with the identity half (#1310, #1639, #1335) and the assistant direct test that landed tonight. Everything here is committed and verified except one arm flagged PENDING below (the base-model paired-story round, run today, harvest pending).

## Motivation

* The 2026-07-14 update left a tension: the assistant context→answer map holds up in plain "User:/Assistant:" transcripts with no chat template, but collapses for generic stories ([#825](https://eps.superkaiba.com/tasks/825)) — even though plain transcripts look superficially story-like.
* That tension splits into two questions:
    * **Framing:** why does the mapping hold in "User:/Assistant:" format but not in stories?
    * **Identity:** is the map about the assistant specifically, or does any consistent speaker carry the same map once the framing is right?
* The original story comparison had confounds that could each explain the split:
    * equal predictive R² across formats does not mean the same *operator*
    * the stories were a different corpus with arbitrary characters, not the assistant persona
    * story activations were read at one position, chosen by convention
    * the story answers were captured teacher-forced
* I ran ablations removing each confound, then the mirror-image identity tests: the same conversations rendered in three framings, operator-identity tests, a read-position sweep, and an on-policy story arm ([#1345](https://eps.superkaiba.com/tasks/1345)); per-character fiction maps ([#1310](https://eps.superkaiba.com/tasks/1310)); a cross-character operator-similarity battery ([#1639](https://eps.superkaiba.com/tasks/1639)); a direct assistant-vs-character test on row-paired cells; register/addressee ablations ([#1417](https://eps.superkaiba.com/tasks/1417)); real published fiction ([#931](https://eps.superkaiba.com/tasks/931)); an identity-vs-framing attribution ladder ([#1335](https://eps.superkaiba.com/tasks/1335)).

## TLDR

- **There is one context→answer operator, attached to the turn-structured speaker-label + response-slot format.** The chat template, the register, and the speaker's identity turn out to be nearly irrelevant to it; narrative framing is what breaks it.
- **Plain "User:/Assistant:" text carries the same operator as chat, expressed in different linear coordinates:** naive cross-format transfer falls short (mean deficit −0.10 instruct / −0.43 base), but a fitted general-linear change of coordinates recovers each ceiling to within 0.005–0.008 (nulls −0.02 to −0.03); instruction tuning pulls the coordinates together (raw operator cosine 0.651 vs 0.293 base, aligned 0.855 vs 0.732)
- **Narrative framing collapses the map on *identical* conversations — a framing effect, not a corpus effect:** verbatim-embedded stories read R² **−0.31** vs chat **+0.24** on the same 2,163 rows; on-policy stories **−0.55** vs **+0.53** (2,018 rows), so teacher-forcing is not the cause. The collapse is partly positional, but no read position rescues it: the best slot rises to −0.02 (CI straddling zero) while the max-over-slots deficit vs chat stays wholly negative (−0.26)
- **The failure is confined to the story context representation:** the story operator reparameterized into chat recovers **0.61 / 0.56** against matched-row chat ceilings of 0.24 / 0.53, and chat contexts predict the story answers at 0.56 — only the direction into story coordinates fails
- **Speaker identity barely moves the operator:** across four fixed-label fiction characters, one pooled map with a single global offset recovers **81–98%** of each character's own-map ceiling, and the character-specific remainder is a small slope residual (+0.007 to +0.025). Character maps align at data-paired Procrustes cosine 0.516 (base) / 0.593 (instruct), between the story↔chat pair (0.455) and the base↔instruct pair (0.686). That ordering carries no CI on the gaps, but the direct manipulations say the same thing with magnitudes: renaming the responder moves the fiction map by at most 0.009
- **The assistant sits inside the shared operator (both arms landed):** on row-paired cells answering the same 4,045 real user questions, assistant↔character reparameterization at fixed format recovers **95–97%** of ceiling in both models; the weak direction is again fiction-framed → Q&A (81–84%); identity-only Procrustes reads 0.540 base / **0.740** instruct — on instruct, above the base↔instruct anchor
- **Base model: same structure, weaker and noisier.** Pooled character fractions 0.81–0.92; coordinates much further apart (raw operator cosine 0.293 vs 0.651; naive transfer deficit −0.43 vs −0.10). Off-template the base reaches ~93% of instruct strength (0.578 vs 0.625; ~83% with the template) — what instruction tuning mostly does is standardize coordinates for an operator the base model already carries

**So the answer to the motivating question:** the discriminating variable is whether the context ends in a turn-structured speaker-label + response slot, not how story-like the surface text looks and not who is speaking. Plain "User:/Assistant:" keeps that structure, so it is the same map up to a linear change of coordinates, and a fiction character answering inside that structure carries nearly the same map too. Narrative prose genuinely moves the context into coordinates no linear map reads, even though the one-way recovery and the answer-side read say the operator and the answer information are still in there.

## Methodology

- **Models:** Qwen2.5-7B (pretrained base) vs Qwen2.5-7B-Instruct
- Same rig throughout: $v_C$ = activation at the end of the context, $v_A$ = mean activation over the answer span, closed-form ridge (GCV-selected penalty), 5-fold conversation-grouped CV, held-out R², layer 19 headline, conversation-level shuffle nulls + bootstrap CIs
- **Framing half:** three framings of the same assistant persona — chat template / plain `User: … Assistant: …` / narrative prose in which a character named "Assistant" answers a human character's spoken question. Chat and no-template use the parent's 4,724 LMSYS single-turn conversations, answers generated on-policy by the measured model; paired stories re-render 2,163 of those same conversations with the original answer embedded verbatim (capture teacher-forced at the fixed answer span); on-policy stories re-render 2,018 of them with the story character answering freely in its own words (one sample per conversation, temperature 1.0)
- **Identity half:** four fixed-label story characters (a warm helper, a calm AI, an ordinary person, a theatrical villain), ~300 scenes each, on-policy stories generated per model, context→that-character's-dialogue ridge maps per persona and model; the assistant test row-pairs three cells on the same 4,045 real user questions — assistant bare Q&A / a fixed character ("Wren") bare Q&A / the same character inside a fiction scene — answers on-policy per cell
- **Operator-identity tests, shared across both halves:** naive cross-cell transfer of a frozen map; data-paired Procrustes-aligned operator cosine against rotation AND shuffled-fit nulls; general-linear reparameterization $A \cdot M \cdot B$ (fitted linear input/output alignments around a frozen operator) against matched-capacity nulls; and a pooled-map lattice (M0 one shared map + global offset / M1 + per-cell offsets / M2 dedicated per-cell maps)

## Results

### _Result 1: "User:/Assistant:" is the same operator as chat, in different linear coordinates_

Equal R² across formats does not establish the same map. The first test is operator identity, run directly on the chat / no-template pair.

**Methodology (this result):**
- Fit the map separately per format on the shared 4,724 conversations (identical folds)
- Apply each format's frozen map in the other format (naive transfer), then retry with fitted linear input/output alignments around the frozen map (reparameterization), against matched-capacity nulls
- Compare the operators directly: raw cosine and Procrustes-aligned cosine vs a rotation null

![reparameterization recovery per direction, context arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/319fd10131dd2517a4099755153efe500dec9e5b/figures/issue_1345/reparam_recovery_context.png)

![raw vs rotation-aligned operator cosine, instruct and base](https://raw.githubusercontent.com/superkaiba/explore-persona-space/319fd10131dd2517a4099755153efe500dec9e5b/figures/issue_1345/operator_cosine_raw_vs_aligned_context.png)

**Takeaways:**

- Naive transfer misses each ceiling (mean per-direction deficit −0.103 instruct / −0.430 base, bootstrap CIs excluding zero) — the two formats are **not** literally the same coordinates
- But a general-linear change of coordinates recovers each format's full ceiling: recovered R² lands between 0.005 below and 0.008 above the target ceiling in all four cells (matched-capacity nulls −0.02 to −0.03). One operator, two coordinate systems
- Instruction tuning canonicalizes the coordinates: raw operator cosine 0.651 (instruct) vs 0.293 (base), aligned 0.855 vs 0.732 (rotation null ≈ 0.001)
- Register and addressee do not matter either ([#1417](https://eps.superkaiba.com/tasks/1417)): rude, evasive, addressee-free, and AI-relay framings all graded Shared against the chat reference at recovery fractions 0.71–1.00
- The single-turn no-template numbers pending in the last update also landed ([#825](https://eps.superkaiba.com/tasks/825)): instruct 0.625 no-template vs 0.654 chat; base 0.578 vs 0.542 — the template helps instruct slightly and *hurts* the base slightly

### _Result 2: Narrative framing collapses the map on identical conversations — teacher-forced and on-policy_

The parent's story result used a different corpus, so I re-rendered the *same* conversations as narrative stories, first with the original answers embedded verbatim, then with the story character answering freely.

**Methodology (this result):**
- 2,700 of the 4,724 shared conversations rendered by the instruct model as short narratives in which a person asks the original question aloud and a character named "Assistant" replies with exactly the original words (`Assistant replied: "…"`); 2,163 survive the verbatim-embedding + judge gates; capture teacher-forced at the answer span
- An on-policy companion at powered n: 2,018 conversations where the story character answers freely in its own words (no reference answer anywhere in the generation prompt)
- Matched-row chat and plain-text refits on the same conversations as comparators; verdict = the story cell's conversation-bootstrap CI (wholly below zero = framing effect, wholly above = corpus effect)

![same conversations under story, chat, and plain-text framings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2390b30a5e37e483bdb0d2c41ea40a3026c22edd/figures/issue_1345/conversation_paired_stories_assistant/framing_effect_matched_rows_bars.png)

![per-conversation relative error, chat map vs on-policy story map](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a3b734c51256fde249d7f08d1b14b8b2981e4dd9/figures/issue_1345/onpolicy_assistant_story/framing_effect_per_conversation_scatter.png)

**Takeaways:**

- Both arms give the pre-registered framing-effect verdict: verbatim-embedded stories read R² **−0.31** (CI −0.33 to −0.28) vs chat **+0.24** on the same rows; on-policy stories read **−0.55** (CI −0.57 to −0.52) vs chat **+0.53** — per conversation the story map is worse for 88% / 98.6% of rows
- The matched-row ceilings sit below the full-corpus values because the story gates keep a selected subset: chat drops from 0.65 to 0.24, and plain text drops much harder, to 0.02 on the teacher-forced round's rows. Why selection punishes the template-free fit by an order of magnitude more than chat is unexplained in this round. Either way, every story-vs-chat comparison here is within the same row set
- Teacher-forcing does not create the collapse: it inflates the story read by +0.075 on matched rows, and the on-policy run is more collapsed (−0.55 vs −0.31)
- The story maps still beat their shuffle nulls (≈ −2.0) — a residual weak signal exists, but the map predicts worse than the answer mean at every layer

### _Result 3: The collapse is partly positional, but no read position recovers the chat ceiling_

Was the collapse just a bad read slot? The anchor read the story context at one conventional position, the colon in `Assistant replied:`. The sweep probes four more.

**Methodology (this result):**
- Four additional context read positions per story — end of the question utterance, last token before the attribution sentence, last token fully before the answer span, and a mean-pool over the attribution phrase — all captured in one teacher-forced forward over the same 2,163 stories (answer-span overlap 0.0 by construction)
- Verdict statistic D = the per-draw max over slots of the paired deficit vs the same-rows chat comparator (1,000 conversation-grouped bootstrap draws; per-slot CIs Bonferroni-adjusted)

![per-slot within-story R² with the chat ceiling band](https://raw.githubusercontent.com/superkaiba/explore-persona-space/38625f8dbcd99b8e120bee1d6551b4a44ff02be6/figures/issue_1345/story_slot_ablation/slot_hero_l19_bar.png)

**Takeaways:**

- The anchor's extreme read was partly the slot: the last token before the answer rises from the anchor's −0.31 to **−0.02** (adjusted CI −0.04 to +0.01, straddling zero); the other slots read −0.26 to −0.40
- But no position recovers chat: the max-over-slots deficit D = **−0.259** (CI −0.276 to −0.243), wholly below zero — at best the story context carries *no* linear map to the answer at any probed position

### _Result 4: The failure sits in the story context coordinates; the operator itself reparameterizes into chat_

If the operator is shared and only the coordinates differ, the story-side operator should recover the chat ceiling once linearly re-expressed. That is the Result-1 reparameterization, run here on the story/chat pair.

**Methodology (this result):**
- Same data-paired reparameterization as Result 1, both directions, on the paired (teacher-forced) and on-policy story rounds; shuffled-fit and rotation nulls

![reparameterization per direction, story and chat pair](https://raw.githubusercontent.com/superkaiba/explore-persona-space/45ed13728538b33de8d727b4e1b875c85f963bcf/figures/issue_1345/conversation_paired_stories_assistant/reparam_r1_r4_direction_points.png)

**Takeaways:**

- The reparameterization is one-way: the story operator moved into chat recovers **0.61** against the matched-row chat ceiling of 0.24 (teacher-forced) and **0.56** against 0.53 (on-policy), while the chat operator moved into story coordinates fails (−0.17 teacher-forced; +0.16 on-policy, weakly positive at best). Landing above the matched ceiling is partly a two-stage-ridge composite effect (the fitted alignments add capacity), so I read the 0.61 as the story operator carrying the chat map's content rather than genuine super-recovery
- The answer side is intact: chat contexts predict the story answers at **0.56** — the story answer representation is a normally predictable target; the failure is specifically that nothing linearly reads the story *context* slot
- I read this as one shared operator whose story-side context coordinates carry framing-specific variability the ridge cannot see through (the alignment asymmetry — context alignments +0.25 story→chat vs −0.26 chat→story, answer alignments ~0.78 both ways — is the errors-in-variables signature)
- Real published fiction is worse still ([#931](https://eps.superkaiba.com/tasks/931)): the chat map transfers to novels at 5% of ceiling or less, and the apparent novel map is mostly an author-level component

### _Result 5: A fixed-label fiction character supports a real but much weaker per-character map_

The identity half starts with the cleanest story-side analog: one fixed character with a stable name, many stories, one map per character ([#1310](https://eps.superkaiba.com/tasks/1310)). The parent had pooled across arbitrary characters.

**Methodology (this result):**
- 4-persona panel, on-policy stories generated per model with the character always under the same label; context→that-character's-dialogue ridge maps, per persona, base and instruct
- Character-swap control (fit on character A, test on character B's dialogue) and scene-level aggregation (one point per scene)

![per-turn vs one-point-per-scene held-out R² at layer 19](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fe0f5271b3bda839da13dd00ec7b00060ae12c0e/figures/issue_1310/agg_vs_perturn_l19.png)

**Takeaways:**

- Scene-aggregated, all four personas clear their shuffle nulls in both models: base **0.13–0.22**, instruct **0.23–0.40**
- The map is character-specific at the raw-map level: the correct pairing beats the character swap by pooled R² **+0.30** (base) / **+0.38** (instruct)
- Grain caveat: aggregated R² is not comparable to the per-turn assistant ceiling (0.588 base / 0.673 instruct) — averaging ~5–6 completions per scene mechanically raises attainable R²
- Absolute levels undersell the maps: the assistant map itself reads about 0.45 when fit on 250 conversations (0.673 at 5,000; the character cells have n≈300), and at matched n and rig the genuine fiction deficit is about **0.12–0.18** of R², attributed to the scene framing ([#1335](https://eps.superkaiba.com/tasks/1335))

### _Result 6: The four character maps are one dominantly shared operator; direct identity manipulations barely move it_

Four per-character maps say nothing yet about whether they are four different operators. The similarity battery across the characters ([#1639](https://eps.superkaiba.com/tasks/1639)) answers that.

**Methodology (this result):**
- Shared-vs-specific decomposition lattice per model: M0 (one pooled map + one global offset) / M1 (+ per-character offsets) / M2 (dedicated per-character maps), scenario-bootstrap CIs
- All 24 ordered raw cross-applications; data-paired reparameterization per ordered pair; data-paired Procrustes-aligned operator cosine with shuffled-fit AND rotation nulls, placed on a single scale against calibration anchors (story↔chat from Result 4; base↔instruct from #825)

![shared-vs-specific decomposition lattice per persona and model](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9edaab4fa46a7bd10be7a0fcaae8d2aa8d2760b5/figures/issue_1310/xpersona_decomposition.png)

![operator similarity statistics with nulls and calibration anchors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7e367f9a5a6a7bd07583aa5595fb28fe4944ee25/figures/issue_1310/xpersona_cosine_reparam.png)

**Takeaways:**

- One pooled map + one global offset recovers **81–98%** of each character's own-map ceiling (base 0.81–0.92, instruct 0.90–0.98); per-character offsets buy nothing (M1−M0 CIs straddle zero); the genuinely character-specific remainder is a small slope residual (M2−M1 = +0.007 to +0.025, every CI above zero, largest for the villain)
- Raw maps do not cross-apply at all (all 24 off-diagonal transfers negative, −0.22 to −2.6). The block sits in the coordinates rather than the operator: a learned linear alignment around a frozen source map recovers **84–97%** (instruct) / **60–79%** (base) of each target ceiling, vs matched-capacity nulls ≈ −0.02
- Data-paired Procrustes-aligned cosine reads **0.516** base / **0.593** instruct (shuffled-fit null ≈ 0.002), between the story↔chat pair (0.455) and the base↔instruct pair (0.686). The anchor ordering carries no CI on the pairwise gaps (0.516 sits close to 0.455), so on its own I read it as consistent with framing moving the operator more than character identity; the direct manipulations below carry the magnitudes
- One correction along the way: an earlier "aligned cosine ≈ 0.99" was retracted as empty — that statistic (a two-sided Procrustes optimum) compares only singular-value spectra, and maps fit on deliberately scrambled pairings under the same ridge recipe also score ~0.99 on it; the read that survives the scrambled-fit null is the data-paired cosine above
- Direct identity manipulations agree ([#1335](https://eps.superkaiba.com/tasks/1335)): renaming the responder on identical text moves the fiction map by at most 0.009, and generating as a character literally named "Assistant" lifts it by at most +0.046; the fiction scene framing accounts for 0.14–0.16 of the 0.17 assistant-vs-fiction gap at matched n

### _Result 7: The assistant itself sits inside the shared operator_

The character battery says fiction characters share one operator. The last cell to place is the assistant itself: the same battery, with the assistant as one of the row-paired cells.

**Methodology (this result):**
- Three row-paired cells answering the same 4,045 real user questions, per model: assistant bare Q&A / a fixed character ("Wren") bare Q&A / the same character inside a fiction scene; answers on-policy per cell; per-question grain
- Same lattice / transfer / reparameterization / Procrustes battery as Result 6; the three pairwise contrasts isolate identity-only (assistant vs Wren, fixed format), framing-only (Wren bare vs Wren-in-scene, fixed identity), and the headline (assistant vs character-in-fiction)

![assistant direct test: lattice and operator similarity, both models](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9e65fe09adad49255a47e2373a84abfdd61c8dfb/figures/issue_1310/xpersona_assistant_test.png)

**Takeaways:**

- Pooling the assistant with the character cells recovers **83–87%** (base) / **81–86%** (instruct) of each cell's own ceiling — the same band the character↔character pooling reads
- Identity costs almost nothing: assistant↔character reparameterization at fixed bare-QA format recovers **95–97%** of ceiling in both models, and on instruct even the *naive* cross-identity transfer keeps 83% of ceiling
- The weak direction is again out of the fiction frame (**81–84%** recovered; every raw fiction-direction transfer negative) — the same story-side context distortion as Result 4, now visible at the operator level
- Identity-only Procrustes cosine reads 0.540 (base) / **0.740** (instruct) — on instruct, *above* the base↔instruct anchor (0.686), so swapping assistant→character moves the operator less than swapping base→instruct does; the two fiction-frame pairs read 0.42–0.49, right in the framing band

## Caveats

- MODERATE confidence throughout: one model family (Qwen-2.5-7B), single seed per cell in most rounds; the base fiction endpoint swings across independent generation runs (0.269–0.435), so read every base attribution as a bound
- A corpus-provenance correction is in progress: the shared single-turn answer corpus is instruct-generated, and both models' chat/plain comparator cells teacher-force that same text — base cells in this line are therefore base-over-instruct-text. Contrasts stand (the text is matched across framings), but absolute base levels carry an off-model-text caveat
- The story-collapse verdict is instruct-only for now. **PENDING:** a base-model paired-story round holding the answer text fixed ran on GCP today (base cannot write the required stories itself); its harvest and verdict fold into #1345 when it lands
- Reparameterization recoveries above a target's own ceiling read as full recovery rather than genuine super-recovery: the pipeline stacks two ridge fits whose combined regularization can out-generalize a single direct fit

## Next Steps

- Fold the pending arms: the base paired-story verdict into #1345 (closing the "base N/A — not tested" scope caveat) and the assistant-test section into #1639
- Localize what breaks in the story context coordinates. The collapse sits in the context representation at every probed position, so a token-position / component ablation of $v_C$ should say *which* content the story coordinates scatter
- Nonlinear probe on the story cells: all story fits are ridge-only; the two-turn result showed the MLP is format-invariant where ridge fails, so an MLP on the story context slot would say whether the collapse is linear-only
- What the residual fiction-frame deficit is made of ([#1335](https://eps.superkaiba.com/tasks/1335)): identity is settled as nearly free, so the remaining 0.12–0.18 at matched n, the cost of the scene framing itself, is the open question

## Pointers

Clean results: #1639 (shared character operator), #1310 (per-character maps + the assistant test artifacts), #1345 (framing collapse), #1417 (register/addressee invariance), #931 (real fiction), #825 (the assistant map, inherited from pretraining), #1335 (identity/framing attribution ladder). Dashboard: https://eps.superkaiba.com/tasks/1639 and siblings. Predecessors merged into this doc: `docs/mentor_updates/2026-07-21_framing_ablations_results.md` (framing half) and the 2026-07-14 update (the parent line).
