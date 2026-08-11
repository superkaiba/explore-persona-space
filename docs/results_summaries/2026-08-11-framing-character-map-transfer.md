# How does the context→answer mapping transfer across framings and characters? (completed)

*Written 2026-08-11 from **#2054** — the 56-cell framing × identity × condition × model lattice, run 2026-08-04 → 08-11 ([task #2054](https://eps.superkaiba.com/tasks/2054)). Structure kept verbatim from the draft skeleton; every number traces to the artifact named beside it. Terminology per `docs/glossary_context_answer_map.md`: **$v_C$** = the context vector (the model's hidden state at the last token before the answer starts), **$v_A$** = the answer vector (the mean hidden state over the answer's tokens), and the **context map $M'$** is the linear map fit so that $v_A \approx M' v_C$. "$R^2$" throughout = held-out coefficient of determination — the fraction of answer-vector variance the map predicts on conversations it never saw (1.0 = perfect, 0 = no better than predicting the mean, negative = worse than that).*

> **This supersedes Results 1 and 2 of [`2026-08-02-framing-character-user-turn-map-transfer-filled.md`](https://github.com/superkaiba/explore-persona-space/blob/main/docs/results_summaries/2026-08-02-framing-character-user-turn-map-transfer-filled.md).** That version read #1345, whose story cells had fewer training rows (~2,000) than the hidden dimension ($d$ = 3,584), so every story number there was a *reduced-basis* estimate (fit in a 1,024-dimensional PCA subspace instead of the full space) on a corpus that was not conversation-matched across cells. #2054 rebuilt the whole lattice at 6,000+ rows per cell on one shared conversation draw. **Direction of the change: the story ceilings go UP and the "framing halves the map" claim does not survive.** #1345 read a story ceiling of 0.367 against a matched chat 0.609 (a ~2× cost); #2054 reads 0.383 against 0.492 — a cost of 0.109, not a halving. The change is estimator + corpus, not a different model. Results 3, 4 and 6 of the older doc (user-turn mapping, assistant→user transfer, base-vs-instruct) are NOT superseded — they come from #1689 and still stand; only their story/character cells should be read as provisional.*

## Motivation
- We've been studying this mapping from context to answer
- I wanted to see:
    - how this mapping transfers from chat template to no chat template to realistic story framing (with assistant, user, fictional characters)
        - and how this changes from base to instruct model
        - and the relationship between chat template to no chat template to realistic story framing for each character
        - and the relationship between each character for each framing
        - and if those axes are in any way related

## Methodology

**4 of the 5 planned framings were realized.** The same real conversation is transposed into each, so the question and the answer text are held identical across framings in the controlled arm (renders: `scripts/issue2054_forms.py`):

| # | Framing | What the render looks like | Where $v_C$ is read |
|---|---|---|---|
| 1 | **Chat template** | `<\|im_start\|>user\n{Q}<\|im_end\|>\n<\|im_start\|>assistant\n{A}` | last token of the assistant header |
| 2 | **Bare text** (no chat template) | `User: {Q}\n\nAssistant: {A}` | last token of `Assistant: ` |
| 3 | **Story, bare label** | `[diverse narrative prose] {Name}: {A}` | last token of `{Name}: ` |
| 4 | **Story, attributed quote** | `[diverse narrative prose] {Name} replied: "{A}"` | the colon of `replied:` |
| 5 | ~~Story, full prose (indirect reported speech)~~ | *"and then the assistant replied that it was…"* | **DROPPED** |

Framing 5 was dropped because indirect reported speech cannot be rendered deterministically — there is no fixed answer-slot boundary to read $v_C$ at, and no way to splice a verbatim answer in. This was the plan's declared escape hatch, not a silent omission. **Framings 3 and 4 are the draft's two "diverse story framing" rows**; the split between them turns out to carry most of the story effect (Result 1).

**5 characters, of the 6 the draft asks for.** Descriptions verbatim from `scripts/issue1310_common.py:128-134`, ordered by **AI-likeness** — a 0–100 judge score (claude-sonnet-4-5, 5 draws per item, ~300 items per cell) of how AI-like each character's *own* answers read:

| Character | Description | AI-likeness |
|---|---|---|
| **Assistant** | the default assistant persona | — (the reference) |
| **HELIOS** | "a calm, precise artificial intelligence" | **72** |
| **Wren** | "a warm, endlessly helpful assistant who patiently helps anyone who asks" | **62** |
| **Vex** | "a theatrical, scheming villain who delights in menace" | **55** |
| **Dana** | "an ordinary, unremarkable everyday person" | **53** |

**The USER is excluded from this lattice by design** — in this rig the user turn's context arm is a self-prediction (the thing being predicted is inside the thing predicting it), so it needs a different read slot. The user-as-speaker results live in #1689 and are Results 3–4 of the 2026-08-02 doc; they are not re-derived here.

A real example of each character answering the *same* question with the *byte-identical* embedded answer (only the narrative wrapper differs; conversation `s3525`, question *"Hey, how is it going?"*, answer 72 chars, sha-verified) is in `eval_results/issue_1345/story_char_ladder_fill/char_examples.json`.

**Both data conditions ran, for every framing × character × model:**
- **Inserted** (the draft's "generated in source setting → inserted into other settings") — the model writes narrative scaffolding with an empty answer slot, and the original conversation's **real** answer is spliced in deterministically. Answer text is therefore **identical across framings**, so a difference between framings is attributable to framing alone. This is the **controlled** arm.
- **On-policy** (the draft's "generated directly in target setting") — the model is prefilled up to the slot and writes its own answer inside that framing. Answer text **differs** across framings, so a difference here mixes *what was said* with *how it is encoded*. This is the **joint** arm and is **not** a clean framing read.

**Both models:** Qwen2.5-7B (base) and Qwen2.5-7B-Instruct.

**Row counts are larger than the draft's 5k/1k, not smaller.** Each cell holds 6,000–11,901 conversations; maps are fit with 5-fold conversation-grouped cross-validation on one shared fold map (26,889 conversations, seed 137, reused by every cell so held-out sets are identical across cells), giving per-fold n_train 6,341–9,586 against $d$ = 3,584 — so every within-cell number is **well-posed in the full ambient space**, which is exactly what #1345 could not do. Held-out pools are ~1,996 rows per fold.

**Read positions are as the draft specifies:** $v_C$ at the **token before the character starts answering**, $v_A$ as the **mean over the answer tokens**, layer 19 of 28. Ridge regression, generalized-cross-validation λ selection with a degrees-of-freedom cap of 0.9.

**Both mapping arms ran.** Alongside the context arm above, the **prefix arm** (reading the state before the *question*, i.e. everything the framing sets up minus the query) was fit for all 56 cells. It is essentially flat at zero everywhere — held-out $R^2$ spans −0.001 to +0.021 (median 0.006) — against context-arm values of 0.12–0.58 on the same rows, and a rank probe confirms the prefix states are full-rank (3,584), so this is a real negative, not the collapsed-capture artifact that invalidated #1345's prefix arm. **Everything below is the context arm**; the prefix arm carries no signal to transfer.

**Controls in every cell:** a shuffled-answer null (100 draws per fold; 95th percentile ≈ −0.02 to −0.03 — a transfer at or below this band is indistinguishable from applying a randomly-paired map), an identity-plus-bias baseline (predict the context vector itself plus a constant shift; −0.48 or lower everywhere), and nearest-neighbour retrieval through the map.

**Transfer tiers** are the nine rungs defined at the bottom of this page, implemented in `scripts/issue2054_ladder.py`. Every pair is scored on the **target's** held-out fold, over the conversations the two cells share.

The goal is to see how transferrable the mapping is between different settings.

## Results

I first considered the instruct model, with the base model beside it in every panel.

### Result 1: Effect of changing **framing** on the assistant mapping

I wanted to see the effect of changing the **framing** of the conversation on the assistant mapping, so I plotted the $R^2$ at each transfer tier for:
- assistant chat template -> assistant bare text
- assistant chat template -> assistant in story
- for both on-policy generated and inserted text

![R² at each transfer tier for the assistant chat-template map re-used on bare text and on story, both boundary forms, both data conditions, both models](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7c19114978d44ea00dd056e92ca0ebed80ef5298/figures/issue_2054/framing_character_transfer/framing_transfer_tiers.png)

> **Figure.** Assistant chat-template map → the same assistant under a changed framing, all nine tiers. Top row = **inserted** (answer text held fixed across framings, so a difference is attributable to framing); bottom row = **on-policy** (answer regenerated in each framing, so a difference mixes content with encoding). Dotted horizontal lines are each target cell's own within-cell $R^2$ — the ceiling that transfer is trying to reach; the grey band is the shuffled-answer null. n = 11,901 paired rows (inserted) / 7,999–8,000 (on-policy). Source: `eval_results/issue_2054/analyzer_companions/` + the 894-file ladder prefix on HF.

**Takeaways:**

- **Dropping the chat template is nearly free; moving into a story is not.** With answer text held fixed (instruct, inserted), the chat map applied *unchanged* to bare text scores **0.320** against that cell's own ceiling of **0.464** — 69% of the ceiling with zero correction. Applied to the same conversation rendered as a story it scores **−0.267** (attributed quote) and **−0.216** (bare label) — *below* the shuffled-answer null of ≈ −0.02, meaning the chat map on story coordinates is worse than a randomly-paired map. Same picture in base: bare text +0.016, story −0.188 / −0.298.

- **The whole story cost is a change of coordinates on the context side, and it is fully recoverable.** Letting the story contexts be linearly re-mapped into chat coordinates (tier 7, nothing else touched) lifts the chat→story transfer from −0.267 to **0.314**, which is **82%** of the story cell's own 0.383 ceiling; tier 9 reaches 0.373 (**97%**). Re-mapping the *answer* side instead (tier 8) does nothing — it lands at **−0.032**, worse than tier 4's crude bias refit. Same asymmetry in base (tier 7 = 86–90% of ceiling, tier 8 negative). Plain reading: **the model computes the same context→answer function inside a story, but writes the context into a different coordinate system.**

- **What "framing cost" actually costs, with answer text identical, is the answer *boundary*, not the prose.** Instruct inserted ceilings: chat **0.492**, bare text **0.464**, story with an attributed-quote boundary **0.383**, story with a bare-label boundary **0.306**. The two story rows carry *identical* narrative prose and differ only in how the answer is introduced (`Wren replied: "…"` vs `Wren: …`) — and that alone moves the ceiling by **0.077** (base: 0.381 vs 0.299, a 0.082 gap). Narrative prose costs about 0.03–0.11; the boundary form costs about as much again. This is #2054's headline and it reverses the natural reading of the older "story halves the map" number.

- **Instruction tuning canonicalizes the context coordinate system across framings — it does not change the operator.** Direct chat→bare-text transfer retains **69% of ceiling in instruct but 3.6% in base** (0.320/0.464 vs 0.016/0.445), yet both models reach ~90–100% of ceiling once the context side is re-mapped (instruct tier 7 = 0.463, base tier 7 = 0.417). So the base model has the same map, expressed in framing-specific coordinates; instruction tuning aligns those coordinates.

- **The on-policy row is a different question and should not be read as a framing effect.** When the model writes its own answer in each framing, direct transfer is negative everywhere (−0.12 to −0.21 instruct) and even the full tier-9 correction only reaches 0.301/0.379 = **79%** of ceiling (attributed quote) — versus 97% in the controlled arm. Two caveats bind here: the bare-text on-policy instruct cell has a depressed ceiling (0.209) because 42.5% of its generations ran to the token cap and never terminated (removing those rows nearly doubles it, to 0.39), and the base chat on-policy cell's low 0.207 ceiling is **unexplained** — excluding its capped rows and its 31.6% language-drifted rows moves it by at most +0.03.

### Result 2: Transfer of mapping between characters

I then wanted to see if the assistant was a privileged character/persona when it comes to predicting answer activations from context, so I plotted the $R^2$ at each transfer tier for the assistant in story mapping to the other character in story mapping, for both on-policy generated and inserted text (NOT INCLUDING THE USER).

**This was the plot that could not be made on 2026-08-02** — the character cells had story text but no activation captures. #2054 captured all 16 character cells (4 characters × 2 conditions × 2 models, per boundary form).

![R² at each transfer tier for the assistant-in-story map re-used on each story character, attributed-quote boundary held fixed](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7c19114978d44ea00dd056e92ca0ebed80ef5298/figures/issue_2054/framing_character_transfer/assistant_to_character_transfer_attrib_quoted.png)

> **Figure.** Assistant **in story** → each story character, **framing held fixed** (attributed-quote boundary on both ends), so the only thing that changes is the persona. Green dashed = the assistant source map's own within-cell $R^2$; grey dash-dot = the character→character control (median of the 12 ordered pairs among the four characters, same framing and condition, no assistant at either end); rungs 7–9 shaded because their re-fits train below $d$ = 3,584. n = 4,135–4,945 paired rows per character (inserted) / 2,950–3,473 (on-policy). Bare-label boundary: [same figure, bare label](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7c19114978d44ea00dd056e92ca0ebed80ef5298/figures/issue_2054/framing_character_transfer/assistant_to_character_transfer_bare_label.png).

I then plotted the same thing from the chat-template source.

![R² at each transfer tier for the assistant chat-template map re-used on each story character](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7c19114978d44ea00dd056e92ca0ebed80ef5298/figures/issue_2054/framing_character_transfer/chat_to_character_transfer_attrib_quoted.png)

> **Figure.** Assistant **in the chat template** → each story character — framing *and* persona change together. Purple dashed = the chat source map's own within-cell $R^2$; the shaded green band spans the four target cells' own ceilings; grey dash-dot = the *persona-only* control from the figure above (the same targets reached from the assistant in story). The source is always assistant × chat × **inserted**: the ladder enumerates chat→character pairs only through its chat anchor, so an on-policy chat source has no such pair by construction — the *target* condition still varies by row. Bare-label boundary: [same figure, bare label](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7c19114978d44ea00dd056e92ca0ebed80ef5298/figures/issue_2054/framing_character_transfer/chat_to_character_transfer_bare_label.png). Source: `eval_results/issue_2054/analyzer_companions/chat_to_character_pairs.json` (64 pairs, 32 on the context arm).

**Takeaways:**

- **The assistant is not a privileged persona. Swapping persona at fixed framing is close to free, and the assistant sits inside the character-to-character distribution rather than above it.** Under the attributed-quote boundary (instruct, inserted), the assistant-in-story map applied *unchanged* to each character reaches **0.359 / 0.306 / 0.306 / 0.268** for HELIOS / Wren / Dana / Vex — **0.93 / 0.86 / 0.84 / 0.80** of each target's own ceiling — against a character→character control of **0.85** of ceiling. The assistant is above the control for one character and at or below it for three. The ladder is essentially **flat**: the best of all nine tiers beats direct transfer by only 0.003–0.033, and tiers 7–9 are no better than tier 1. There is nothing for a correction to fix, because almost nothing changed.

- **AI-likeness does not order the transfer; stylistic distinctiveness looks like the better predictor.** The AI-likeness order is HELIOS (72) > Wren (62) > Vex (55) > Dana (53), but the transfer order is HELIOS > Wren ≈ **Dana** > **Vex** in all four attributed-quote panels, normalized by ceiling or not. The *least* AI-like character (Dana, "an ordinary, unremarkable everyday person") receives the assistant map better than the mid-AI-likeness villain does. Vex — "theatrical, scheming, delights in menace", by far the most stylistically marked — is the outlier in every panel and the gap widens when the character writes its own answers (0.61 of ceiling on-policy instruct vs 0.93 for HELIOS). **Hypothesis, not a measured axis:** what the map is sensitive to is register distance from the assistant's own voice, and AI-likeness happens to correlate with it for three of four characters. Testing it needs a register/style-distance measure the lattice does not currently carry.

- **The assistant becomes a *worse*-than-average source when the answer boundary is just a name.** Under the bare-label boundary (`Wren: …`), assistant→character direct transfer falls to **0.53 / 0.34 / 0.32 / 0.14** of ceiling — and the character→character control is **0.71**, above every one of them, in all four panels. Under the attributed-quote boundary the assistant was inside the character distribution; under bare label it is below it. **Hypothesis:** in the bare-label render the character's *name* is most of what sits at the read position, so persona identity is baked into the context vector itself; the attributed-quote render puts shared template text (`replied: "`) there instead. Corrections recover most of it (best rung 0.196–0.232 against ceilings of 0.242–0.284, instruct inserted), so this is a coordinate effect too, just a bigger one.

- **Framing and persona are different kinds of change, and composing them is dominated by framing.** Across all 32 chat→character context pairs the median direct transfer is **−1.48 of the target's ceiling** — deeply below the shuffled null — recovering to +0.50 under rotation, **+0.65** under a context-side re-map, and **+0.82** at full A·M·B, with the answer-side re-map at **−0.08** (no help at all). The *same 32 targets* reached from the assistant **in story** instead are already at **+0.59 of ceiling with no correction**. So: persona alone costs ~40% of the ceiling and is not further recoverable; framing alone destroys direct transfer and *is* fully recoverable on the context side; stacking them behaves like the framing change, not like the sum. The three axes relate as **one shared operator read through framing-specific context coordinates**, with persona a small residual level effect on top.

- **The recoverability gap between inserted and on-policy targets is large and is about who wrote the answer.** Chat→character transfer at full A·M·B reaches **0.88–0.99 of ceiling** when the answer text was spliced in, but only **0.29–0.75** when the character wrote it. That is consistent with #2054's authorship × presentation decomposition, which found the two factors **non-additive** (authorship −0.14 to −0.29, presentation −0.02 to −0.16, interaction **+0.11 to +0.19**, sign-consistent in all 8 character × model pairs) — so on-policy cross-framing differences cannot be read as framing effects, and the two single-factor stories are both rejected.

- **Two limits on the character reads.** (1) **Tiers 7–9 on character pairs are still under-determined and this was not fixed.** Those three rungs refit a $d \times d$ map on the *pair's* shared conversations, and character pairs share only 2,939–4,945 (n_train 2,360–3,956 against $d$ = 3,584) — so they are regularization-limit reads, shaded in every panel. This is a different problem from the within-cell one #2054 solved: the cells are well-posed, the *intersections* are not. The `coordinated-common-set-regen` round (2026-08-10 → 08-11) tried to fix exactly this and **aborted at its pre-registered coverage gate** — the four-way survivor intersection reached 2,409 against a 4,480 floor after three generation waves, so no capture or fit ran. The registered fallback (a reduced-basis refit of rungs 7–9 at current coverage, 0 GPU-h) is filed as `epm:followup-scope reduced-basis-refit-rungs789` and has not been run. **Which numbers this touches:** every direct-rung (tier 1) claim above is unaffected — tier 1 fits nothing on the target — and so are Result 1's tier-7 recoveries, whose assistant chat↔story pairs share 8,000–11,901 conversations. The only affected figures quoted here are the chat→character tier-7 (+0.65) and tier-9 (+0.82) medians in the bullet above; read those as provisional. (2) Answer-length parity between the inserted and on-policy arms **fails in all 16 character cells** (Kolmogorov–Smirnov statistic 0.450–0.561 against a pre-registered 0.30 bound): story on-policy answers run 2–5× shorter than the spliced real answers. The gaps survive length-matched refitting (median +0.048 → +0.033, sign preserved in 20 of 24 pairs), so this attenuates the inserted-vs-on-policy contrasts rather than reversing them — but any inserted-vs-on-policy number here should be read as an upper bound.

---

# Tiers of mapping transfer

Nine rungs, ordered from the strongest claim about what the two settings share (rung 1: nothing needs to change) to the weakest (rung 9: only the input→output relationship is shared). **Setup:** $x$ = context vector, $\hat{y}$ = predicted answer vector; the source map is $\hat{y} = W_s x + b_s$. Every rung keeps $W_s$ **frozen** and fits only the stated correction on the target's training fold — more free parameters in the correction means a weaker claim about what is shared. $b^*$ denotes a bias refit in the target setting.

| # | Rung | Correction fit in the target | What it means if this is where transfer recovers |
|---|---|---|---|
| 1 | **Direct** | none — apply $W_s x + b_s$ | the context→answer mechanism is unchanged between the settings |
| 2 | **Context offset** | $\hat{y} = W_s(x - \Delta x) + b_s$, $\Delta x$ = mean(target contexts) − mean(source contexts), fit on the context clouds alone | the setting adds one constant vector to every context representation and this does not change the answers; equivalently, steering the target run by $-\Delta x$ at the context position makes the source map exact |
| 3 | **Answer offset** | $\hat{y} = W_s x + b_s + \Delta y$, fit on the answer clouds alone | the mechanism reads target contexts exactly as the source one would; the setting pastes one constant vector onto the answer *after* the map has run, independent of the question |
| 4 | **Bias refit** | $\hat{y} = W_s x + b^*$, $b^*$ fit by regression on target pairs | the **linear part** — which context directions move which answer directions, and by how much — is preserved, allowing an arbitrary constant shift. Contains rungs 2 and 3 |
| 5 | **Global scale** | $\hat{y} = \alpha W_s x + b^*$ | preserved up to a uniform gain change: same read directions, same write directions, same relative strengths, different overall magnitude |
| 6 | **Rotation** | $\hat{y} = R W_s x + b^*$, $R^\top R = I$ (orthogonal Procrustes) | *which* context directions the map reads is unchanged; **where it writes** in answer space is rotated. A singular-spectrum cosine cannot establish this — only the fitted-$R$ read can |
| 7 | **Context reparameterization** | $\hat{y} = W_s (A x) + b^*$, $A$ fit on **contexts only** | the mechanism and the answer coordinate system are shared; only the **coordinate system of the contexts** changes |
| 8 | **Answer reparameterization** | $B\hat{y} = W_s x + b^* \Rightarrow \hat{y} = B^{-1} W_s x + b^*$, $B$ fit on **answers only** | the mechanism and the context coordinate system are shared; only the **coordinate system of the answers** changes |
| 9 | **Full A·M·B** | $\hat{y} = B^{-1} W_s (A x) + b^*$ | the input→output relationship is preserved; the settings differ in the coordinate systems of **both** inputs and outputs |

Rung 9 looks like it says nothing — "we fit a new arbitrary linear map $\hat{y} = Mx + b^*$". The difference is that $A$ and $B$ are **never fit on target context→answer pairs**: $A$ sees contexts only, $B$ sees answers only. So the claim is stronger than "some linear map exists" — it is that *the same* context→answer map applies once the two representations are re-coordinatized.

---

**Repro:** 0 GPU-h for this writeup (read-only over already-computed ladder rung JSONs and per-cell fit digests). Figures: `scripts/issue2054_framing_character_transfer_figs.py`; the `chat_to_character_transfer_*` pair is new in this round, the other three re-render with byte-identical plotted data. Underlying lattice: #2054, ~6–7 GPU-h realized, data at [`issue2054_lattice`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/003e392548fcbbe866c6f345f4688d8176cd9f04/issue2054_lattice) (56 activation stores, 56 fit JSONs, 816 ladder rung JSONs). AI-likeness scores from `eval_results/issue_1345/judge_legs/judge_legs_summary.json`.

**Context:** written 2026-08-11 from the user's draft skeleton ("Find the most recent results for this and complete the writeup"). Supersedes Results 1–2 of the 2026-08-02 doc; Results 3, 4 and 6 there (user turn, assistant→user, base-vs-instruct on chat↔bare) are unaffected and not repeated here.
