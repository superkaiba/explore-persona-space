# Result: "User:/Assistant:" is the chat map in different coordinates; narrative framing is what collapses it

## Motivation

* The last update left a tension: the assistant context→answer map holds up in plain "User:/Assistant:" transcripts with no chat template, but collapses for generic stories ([#825](https://eps.superkaiba.com/tasks/825)) — even though plain transcripts look superficially story-like.
* This raises the question: why does the mapping hold in "User:/Assistant:" format but not in stories?
* The original story comparison had confounds that could each explain the split:
    * equal predictive R² across formats does not mean the same *operator*
    * the stories were a different corpus with arbitrary characters, not the assistant persona
    * story activations were read at one position, chosen by convention
    * the story answers were captured teacher-forced
* I ran ablations removing each confound ([#1345](https://eps.superkaiba.com/tasks/1345), [#1310](https://eps.superkaiba.com/tasks/1310)): the same conversations rendered in three framings (chat template / plain "User:/Assistant:" / narrative story whose character is named "Assistant"), operator-identity tests, a read-position sweep, and an on-policy story arm.

## TLDR

- **"User:/Assistant:" is not story-like to the model — it is the same operator as chat, expressed in different linear coordinates:**
    - naive cross-format transfer falls short of each format's ceiling (mean deficit −0.10 instruct / −0.43 base), but a fitted general-linear change of coordinates recovers each ceiling to within **0.005–0.008** (nulls ≈ −0.03), in both models
    - instruction tuning pulls the two coordinate systems together: raw operator cosine **0.651** (instruct) vs **0.293** (base); rotation-aligned **0.855** vs **0.732**
- **Narrative framing collapses the map on *identical* conversations — a framing effect, not a corpus effect:**
    - same conversations, original answers embedded verbatim in a story: R² **−0.31** vs **+0.24** chat on the same 2,163 rows
    - same conversations, the story character answering freely (on-policy, 2,018 rows): **−0.55** vs **+0.53** chat — so teacher-forcing is not the cause (it *inflates* the story read by +0.075)
- **The collapse is partly positional, but no read position rescues it:**
    - sweeping four other context read positions, the best slot (last token before the answer) rises from −0.31 to **−0.02** (adjusted CI −0.04 to +0.01, straddling zero); the max-over-slots deficit vs chat stays wholly negative (**−0.26**, CI −0.28 to −0.24)
- **The failure sits in the story context representation, not the answers — and the operator is still the same one:**
    - the story operator reparameterized into chat recovers **0.61** (teacher-forced) / **0.56** (on-policy) against matched-row chat ceilings of 0.24 / 0.53, and chat contexts predict the story answers at **0.56** — the information and the map survive; the story context slot just is not linearly readable
- **Fiction characters do support a much weaker per-character map:**
    - a fixed-label story character's context→dialogue map clears its shuffle null in both models once each scene is aggregated to one point (base **0.13–0.22**, instruct **0.23–0.40**) and is character-specific (correct pairing beats the character swap by pooled R² **+0.30** base / **+0.38** instruct) — so stories do carry a weak per-character map, far below the assistant one
- **(The pending Result-3 number from the last update also landed:)** instruct single-turn no-template R² = **0.625** (vs 0.654 chat); base **0.578** (vs 0.542 chat) — off-template the base reaches **~93%** of instruct strength (vs ~83% with the template), which strengthens the pretraining-inheritance picture

**So the answer to the question:** the discriminating variable is whether the context ends in a turn-structured speaker-label + response slot, not how story-like the surface text looks. Plain "User:/Assistant:" keeps that structure, so it is the same map up to a linear change of coordinates. Narrative prose genuinely moves the context into coordinates no linear map reads, even though the same operator and the same answer information are still present.

## Methodology

- **Models:** Qwen2.5-7B (pretrained base) vs Qwen2.5-7B-Instruct
- Same rig as the last update: $v_C$ = activation at the end of the context, $v_A$ = mean activation over the answer span, closed-form ridge (GCV-selected penalty), 5-fold conversation-grouped CV, held-out R², layer 19 headline, conversation-level shuffle nulls + bootstrap CIs
- **Three framings of the same assistant persona:** chat template / plain `User: … Assistant: …` / narrative prose in which a character named "Assistant" answers a human character's spoken question
- **Data per arm:** chat and no-template use the parent's 4,724 LMSYS single-turn conversations, answers generated on-policy by the measured model; paired stories re-render 2,163 of those same conversations as instruct-written narrative with the original answer embedded verbatim (capture teacher-forced at the fixed answer span); on-policy stories re-render 2,018 of them with the story character answering freely in its own words (one sample per conversation, temperature 1.0)
- **Operator-identity tests:** naive cross-framing transfer; Procrustes-aligned operator cosine against a random-rotation null; general-linear reparameterization $A \cdot M \cdot B$ (the Result-2 method from the last update) against matched-capacity nulls

## Results:

### _Result 1: "User:/Assistant:" is the same operator as chat, in different linear coordinates_

Equal R² across formats does not establish the same map, so I first tested operator identity directly on the chat / no-template pair.

**Methodology (this result):**
- Fit the map separately per format on the shared 4,724 conversations (identical folds)
- Apply each format's frozen map in the other format (naive transfer), then retry with fitted linear input/output alignments around the frozen map (reparameterization), against matched-capacity nulls
- Compare the operators directly: raw cosine and Procrustes-aligned cosine vs a rotation null

![reparameterization recovery per direction, context arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/319fd10131dd2517a4099755153efe500dec9e5b/figures/issue_1345/reparam_recovery_context.png)

![raw vs rotation-aligned operator cosine, instruct and base](https://raw.githubusercontent.com/superkaiba/explore-persona-space/319fd10131dd2517a4099755153efe500dec9e5b/figures/issue_1345/operator_cosine_raw_vs_aligned_context.png)

**Takeaways:**

- Naive transfer misses each ceiling (mean per-direction deficit −0.103 instruct / −0.430 base, bootstrap CIs excluding zero) — the two formats are **not** literally the same coordinates
- But a general-linear change of coordinates recovers each format's full ceiling: recovered R² lands between 0.005 below and 0.008 above the target ceiling in all four cells (matched-capacity nulls −0.02 to −0.03) — **one operator, two coordinate systems**
- Instruction tuning canonicalizes the coordinates: raw operator cosine 0.651 (instruct) vs 0.293 (base), aligned 0.855 vs 0.732 (rotation null ≈ 0.001)
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

- **Framing effect, both arms:** verbatim-embedded stories read R² **−0.31** (CI −0.33 to −0.28) vs chat **+0.24** on the same rows; on-policy stories read **−0.55** (CI −0.57 to −0.52) vs chat **+0.53** — per conversation the story map is worse for 88% / 98.6% of rows
- The matched-row chat ceilings sit below the full-corpus 0.65 because the story gates keep a selected subset (plain text reads 0.02 on the teacher-forced round's rows) — all comparisons here are within the same row set
- Teacher-forcing does not create the collapse: it inflates the story read by +0.075 on matched rows, and the on-policy run is more collapsed (−0.55 vs −0.31)
- The story maps still beat their shuffle nulls (≈ −2.0) — a residual weak signal exists, but the map predicts worse than the answer mean at every layer

### _Result 3: The collapse is partly positional, but no read position recovers the chat ceiling_

The story cells were read at one conventional position (the colon in `Assistant replied:`), so I swept the read position to check whether the collapse is just a bad slot.

**Methodology (this result):**
- Four additional context read positions per story — end of the question utterance, last token before the attribution sentence, last token fully before the answer span, and a mean-pool over the attribution phrase — all captured in one teacher-forced forward over the same 2,163 stories (answer-span overlap 0.0 by construction)
- Verdict statistic D = the per-draw max over slots of the paired deficit vs the same-rows chat comparator (1,000 conversation-grouped bootstrap draws; per-slot CIs Bonferroni-adjusted)

![per-slot within-story R² with the chat ceiling band](https://raw.githubusercontent.com/superkaiba/explore-persona-space/38625f8dbcd99b8e120bee1d6551b4a44ff02be6/figures/issue_1345/story_slot_ablation/slot_hero_l19_bar.png)

**Takeaways:**

- The anchor's extreme read was partly the slot: the last token before the answer rises from the anchor's −0.31 to **−0.02** (adjusted CI −0.04 to +0.01, straddling zero); the other slots read −0.26 to −0.40
- But no position recovers chat: the max-over-slots deficit D = **−0.259** (CI −0.276 to −0.243), wholly below zero — at best the story context carries *no* linear map to the answer at any probed position

### _Result 4: The failure sits in the story context coordinates; the operator itself reparameterizes into chat_

If the operator is shared and only the coordinates differ, the story-side operator should recover the chat ceiling once linearly re-expressed — so I ran the Result-1 reparameterization on the story/chat pair.

**Methodology (this result):**
- Same data-paired reparameterization as Result 1, both directions, on the paired (teacher-forced) and on-policy story rounds; shuffled-fit and rotation nulls

![reparameterization per direction, story and chat pair](https://raw.githubusercontent.com/superkaiba/explore-persona-space/45ed13728538b33de8d727b4e1b875c85f963bcf/figures/issue_1345/conversation_paired_stories_assistant/reparam_r1_r4_direction_points.png)

**Takeaways:**

- **The reparameterization is one-way:** the story operator moved into chat recovers **0.61** against the matched-row chat ceiling of 0.24 (teacher-forced) and **0.56** against 0.53 (on-policy), while the chat operator moved into story coordinates fails (−0.17 teacher-forced; +0.16 on-policy, weakly positive at best). Landing above the matched ceiling is partly a two-stage-ridge composite effect (the fitted alignments add capacity), so I read it as the story operator carrying the chat map's content, not as super-recovery
- The answer side is intact: chat contexts predict the story answers at **0.56** — the story answer representation is a normally predictable target; the failure is specifically that nothing linearly reads the story *context* slot
- I read this as one shared operator whose story-side context coordinates carry framing-specific variability the ridge cannot see through (the alignment asymmetry — context alignments +0.25 story→chat vs −0.26 chat→story, answer alignments ~0.78 both ways — is the errors-in-variables signature)

### _Result 5: A fixed-label fiction character supports a real but much weaker per-character map_

The parent pooled across arbitrary story characters, so I also tested the cleanest story-side analog: one fixed character with a stable name, many stories, one map per character ([#1310](https://eps.superkaiba.com/tasks/1310)).

**Methodology (this result):**
- 4-persona panel, on-policy stories generated per model with the character always under the same label; context→that-character's-dialogue ridge maps, per persona, base and instruct
- Character-swap control (fit on character A, test on character B's dialogue) and scene-level aggregation (one point per scene)

![per-turn vs one-point-per-scene held-out R² at layer 19](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fe0f5271b3bda839da13dd00ec7b00060ae12c0e/figures/issue_1310/agg_vs_perturn_l19.png)

**Takeaways:**

- Scene-aggregated, all four personas clear their shuffle nulls in both models: base **0.13–0.22**, instruct **0.23–0.40**
- The map is character-specific: the correct pairing beats the character swap by pooled R² **+0.30** (base) / **+0.38** (instruct)
- Grain caveat: aggregated R² is not comparable to the per-turn assistant ceiling (0.588 base / 0.673 instruct) — averaging ~5–6 completions per scene mechanically raises attainable R²; at the per-turn grain the fiction maps stay far below the assistant map

## Next Steps

- **Localize what breaks in the story context coordinates:** the collapse is context-side and not positional — a token-position / component ablation of $v_C$ (still the open Q4 from the last update) should say *which* content the story coordinates scatter.
- **Nonlinear probe on the story cells:** all story fits are ridge-only; the two-turn result showed the MLP is format-invariant where ridge fails, so an MLP on the story context slot would say whether the collapse is linear-only.
- **Base-model paired-story arm:** untested (base story generation yields were too low to run); the framing-effect verdict is instruct-only so far.
- **Why the assistant map is so much stronger than fiction-character maps** ([#1335](https://eps.superkaiba.com/tasks/1335)): the per-character maps exist but sit an order below — what the assistant persona has that fixed fiction characters lack is the open question.
