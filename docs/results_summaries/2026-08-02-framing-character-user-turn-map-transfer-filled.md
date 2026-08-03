# How does the context→answer mapping transfer across framings, characters, and speakers? (filled)

*Filled 2026-08-02 from #1345 (framing/story line, incl. the #1887 estimator-audit corrections and a new same-day ladder-fill round, `eval_results/issue_1345/story_char_ladder_fill/`), #1689 (speaker×framing lattice + user-slot recapture round), and #825 (cross-model anchor). Structure kept verbatim from the draft; every number traces to the artifact named beside it. Terminology per `docs/glossary_context_answer_map.md` ($v_C$ context vector, $v_A$ answer vector, context map $M'$: $v_A \approx M' v_C$). Where a requested cell was never run, that is stated instead of approximated. Summary figures: `figures/results_summaries/framing_character_user_turn/` (script `scripts/results_summary_framing_plots.py`).*

## Motivation
- We've been studying this mapping from context to answer
- I wanted to see:
    - how this mapping transfers from chat template to no chat template to realistic story framing (with assistant, user, fictional characters)
        - and how this changes from base to instruct model
        - and the relationship between chat template to no chat template to realistic story framing for each character
        - and the relationship between each character for each framing
        - and if those axes are in any way related

## Methodology

- I considered **3 different framings**. The same exchange transposed into each (story render verbatim from `data/issue_1345/story_boundary_ablation/stories/kept_stories_bnd_v4_instruct.jsonl` conv s2541; render formats `scripts/issue825_render_formats.py`):
    - **Chat template** — raw special-token render (no default system preamble). Context read at the last token of the assistant header:
      `<|im_start|>user\nWhat's the time now?<|im_end|>\n<|im_start|>assistant\n18:45, or 6:45 PM, according to your device.<|im_end|>`
    - **Bare text (no chat template)** — read at the last token of `Assistant: `:
      `User: What's the time now?\n\nAssistant: 18:45, or 6:45 PM, according to your device.`
    - **Story (narrative prose)** — read at the colon of the attribution `Assistant replied:`:
      *"The sun had just set, casting a golden glow over the quiet park. Mia stood by a bench... 'What's the time now?' she asked, tapping her phone impatiently.\n\nAssistant replied: "18:45, or 6:45 PM, according to your device.""*
      (Character named `ARIA` in round 1, renamed `Assistant` for follow-ups; the rename alone dropped story yield 84.0%→72.2%.)
- I considered **4 different characters** (plus the assistant, and the user as a speaker — Result 3):
    - Descriptions verbatim (`scripts/issue1310_common.py:128-134`), ordered by judge-scored AI-likeness of their own story answers (0–100, Sonnet-4.5 judge, 5 draws/item, 300 items/cell; base-model values, `figures/issue_1345/story_boundary_ablation/character_ai_likeness.meta.json`):
        - **HELIOS** — "a calm, precise artificial intelligence" — most AI-like, **74.2**
        - **Wren** — "a warm, endlessly helpful assistant who patiently helps anyone who asks" — **63.5**
        - **Vex** — "a theatrical, scheming villain who delights in menace" — **57.5**
        - **Dana** — "an ordinary, unremarkable everyday person" — least AI-like, **53.2**
    - Real example of each character's response with the same context — conversation `s3525`, question *"Hey, how is it going?"*, the embedded answer **byte-identical across all four** (72 chars; sha-verified), so only the narrative wrapper differs (`eval_results/issue_1345/story_char_ladder_fill/char_examples.json`):
        - **HELIOS**: *"Helios stood by the window, its screens displaying serene landscapes of distant planets. ... They inquired, "Hey, how is it going?" HELIOS replied: "..."*
        - **Wren**: *"Wren smiled warmly as she arranged a stack of books neatly on a shelf. The bookshop was bustling... "Hey, how is it going?" asked the visitor. Wren replied: "..."*
        - **Dana**: *"Dana had just finished her morning commute and was stepping into the office elevator when a colleague, Jake, caught up with her. "Hey, how is it going?" Jake asked... Dana replied: "..."*
        - **Vex**: *"The dim lighting of the grand hall cast long shadows as Vex, with an extravagant flourish of his cloak, declared his grand entrance. ... "Hey, how is it going?" Vex replied: "..."*
- For each framing I ran 2 conditions (instruct; base story generation missed its yield floor at 96/500 kept and was halted — `eval_results/issue_1345/story_regime_coverage.json`):
    - data generated in source setting -> inserted into other settings (answers generated on-policy in chat, embedded verbatim into the story render; kept 2,164/2,700 — dominant reject: the model paraphrases instead of embedding)
    - data generated directly in target setting (the model writes its own answer inside the story; kept 2,019/3,438 = 58.7%)
- I ran all of the above in both the base and instruct model (Qwen2.5-7B / -Instruct; exceptions stated where they bind: base story cells, and the character cells of Result 2)

Fits: teacher-forced capture at layer 19 of 28, closed-form ridge, conversation-grouped 5-fold CV, shuffled-answer matched-capacity nulls, identity+learned-bias baseline + kNN retrieval per fitted map. Story cells have n_train (1,614–1,730) < d (3,584), where ambient-basis GCV is a known estimator artifact (#1887 audit, replay gate PASS) — **all story-cell numbers below are the corrected reduced-basis reads** (train-fold PCA, k = min(1024, ⌊n_train/2⌋)); the new ladder round reproduces all 8 audited ceilings to 4 dp. Context arm throughout; the #1345 prefix arm is degenerate (activations collapse onto 14 extraction-batch vectors, `eval_results/issue_1345/prefix_degeneracy_probe.json`) — stated scope caveat.

I considered the tiers of mapping transfer at the bottom of this page (the nine metrics there are exactly the implemented rungs 1–9: `scripts/issue1345_ladder_rungs.py`, `scripts/issue1345_story_char_ladder_fill.py`)


I considered the held-out $R^2$ after applying each of these transformations on a fixed training set of generic prompts (LMSYS-derived conversations; corrections always fit on the target's train fold, source operator frozen)

The goal is to see how transferrable the mapping is between different settings

## Results
I first considered the instruct model.
### Result 1: Effect of changing **framing** on the assistant mapping

I wanted to see the effect of changing the **framing** of the conversation on the assistant mapping, so I plotted the $R^2$ at each transfer tier for:
- assistant chat template -> assistant bare text
- assistant chat template -> assistant in story
- for both on-policy generated and inserted text

![R² at each transfer tier: chat→bare (ambient basis, parent ladder round) and chat→story inserted/on-policy (reduced basis, new fill round); dashed = each target's own within-cell ceiling; conversation-grouped 5-fold, L19](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/results_summaries/framing_character_user_turn/r1_framing_ladder.png)

> Sources: `eval_results/issue_1345/ladder_rungs/ladder_rungs_instruct_context.json` (bare, n=4,724; cells #1887-stable so the ambient read stands) + `eval_results/issue_1345/story_char_ladder_fill/ladders.json` (story, reduced basis). Story data conditions per the Methodology; bare-text row is teacher-forced reference answers (no on-policy bare ladder exists).

**Takeaways:**
- **Chat→bare barely needs correcting; chat→story fails until the high tiers.** Direct transfer: bare 0.520 of a 0.625 ceiling (83%); story −0.26 (inserted) / −0.35 (on-policy) — at or below the shuffled-operator null in 3 of 4 story directions, i.e. the uncorrected chat operator is indistinguishable from a randomly-paired one on story coordinates.
- **Where recovery happens is diagnostic.** Into story, context-side reparameterization does nearly all the work (0.302 of the 0.367 ceiling = 82% inserted; 0.178/0.262 = 68% on-policy; full A·M·B 91%/67%); answer-side reparameterization does nothing (−0.12/−0.01). Into chat (reverse, `ladders.json`), rotation is the best single correction (0.472 of 0.609 = 77%) — read against its elevated null (0.14).
- **The story ceiling itself is halved**: 0.367 (inserted) / 0.262 (on-policy) vs matched-row chat 0.609 / 0.567 — framing costs ~2×, generating the answer in-story costs a further ~0.10 (`eval_results/issue_1345/lambda_audit_1887/corrections_table.md`).
- Identity+bias baseline is −0.64 to −0.92 on every story pair while kNN-through-the-map retrieval stays far above chance (`ladders.json` `knn_retrieval_fold0`) — the recurring R²-vs-retrieval dissociation; the maps carry discriminative structure even where pooled R² is low.

### Result 2: Transfer of mapping between characters
I then wanted to see if the assistant was a privileged character/persona when it comes to predicting answer activations from context, so I plotted the $R^2$ at each transfer tier for the assistant in story mapping to the other character in story mapping, for both on-policy generated and inserted text

**This plot cannot be made yet: the character cells have story text but no activation captures.** A three-way relocation sweep (local turnstores, the #1887 staged stores, HF `issue1345_framing/char_*` listings) found 0 of 16 character cells with a capture — only kept stories (2,156–2,187 per character), judge results, and yield reports (`eval_results/issue_1345/story_char_ladder_fill/char_cells.json`, `capture_status: absent` per cell). Unblocking needs one teacher-forced GPU capture pass over the kept stories; the fits themselves are then the same 0-GPU ladder as Result 1. The only existing character-pair ladder (#1689's lattice) is not usable for this question: its story/character cells' own within-cell ceilings are ≈0 (0.001–0.03 instruct), so "transfer" reads there are low-ceiling artifacts — its LOW-confidence headline (same-framing speaker swaps reconcile at tier 1, framing swaps at tier 9 or never) is the pattern, not a measurement.

What the character panel does support today — the judge-scored AI-likeness axis over each character's own story answers:

![Judge-scored AI-likeness (0–100) of each character's own story answers vs the injected verbatim-reference control, base model; ±1.96 SE ≈ 1.7; Sonnet-4.5 judge, 5 draws/item, 300 items/cell](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1345/story_boundary_ablation/character_ai_likeness.png)

**Takeaways:**
- The assistant-privilege question is **unmeasured at the map level** — blocked on a missing GPU capture pass, not on analysis (this round verified the blocker concretely and filed the numbers that do exist).
- Own-answer AI-likeness orders **HELIOS 74.2 > Wren 63.5 > Vex 57.5 > Dana 53.2** (base; instruct 72.1/61.8/55.4/53.1) while the injected control is flat at 78.7–80.8 — the axis reads *who wrote the answer*, not which character is speaking, so character identity shapes generation more than it shapes the judged surface of an identical answer.
- Weak provenance-side hint (#1689 recapture): authoring the assistant's reply as Wren instead of the assistant shifts second-user-turn predictability by ≤0.03 R².

### Result 3: User turn mapping
I then wanted to see if the user character was just another similar character or in some way privileged. For the user turn I considered 3 types of completions:
- Haiku generated
- real user data
- on-policy generated by Qwen

I first plotted the $R^2$ for the user turn mapping in the chat template for each kind of generated data

![Second-user-turn map R² in the chat template per completion provenance, instruct model, clean read slot, L19; solid = turn-end target, light = mean-over-turn target; point estimates (fold spread not persisted in the mirror)](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/results_summaries/framing_character_user_turn/r3a_user_turn_provenance.png)

> Source: `eval_results/issue_1689/user_slot_recapture/summary.json` (recapture round — corrected token boundaries; committed to git by this round from the HF mirror). Three caveats bind: (1) **the "real user data" arm is NOT real user text** — the two-turn LMSYS corpus has no u2 field and `scripts/issue1689_render_conditions.py:309` falls back to the constant string *"Can you say a bit more about that?"* on every row (one distinct sha256 across 2,114 rendered rows); the prefix (u1, a1) is real LMSYS, the predicted turn is a constant. #1689's promoted body still calls this arm "tier-1 real data" — refuted, correction pending with the deferred recapture body-fold. (2) The parent round's on-policy arm was a byte-duplicate of the Haiku arm; only the recapture round is a genuine third arm. (3) Qwen mostly refuses to write user turns — all six on-policy cells missed the 80% yield floor (chat: 2.5% instruct / 4.2% base, n=266/473 rows), so on-policy-chat is partly a sample-size story.

**Takeaways:**
- $R^2$ is **monotone in how constrained the turn is**: constant string 0.69, Haiku-simulated 0.24, on-policy Qwen 0.07 (chat, instruct, turn-end target). No genuine real-user-text arm exists yet.
- The **target convention matters**: mean-over-turn lifts Haiku 0.243→0.374 and on-policy bare-text 0.018→0.287 — much of the apparent unpredictability is the end-token read, not content.
- Framing barely moves the user turn (story ≈ chat ≈ bare per provenance; base same ordering: 0.63/0.27/0.01), and the story speaker label (Alex vs literal "User") is nearly free (Δ ≤ 0.01).

I then looked at if this $R^2$ changed if you ran this at the second user turn (intuitively the first user turn is unpredictable because the model has no prior information about the user)

![First vs second user turn R² per provenance, chat template, instruct, clean read slot (last token strictly before the turn), turn-end target](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/results_summaries/framing_character_user_turn/r3b_first_vs_second_turn.png)

**Takeaways:**
- **The data supports the hypothesis, cleanly.** First user turn: $R^2$ = −0.002 to −0.075 in every cell, both models, all framings — exactly zero with no prior user content. Second user turn: up to 0.24–0.37 (Haiku). Dropping (u1, a1) from the second-turn read costs 0.04–0.09 directly (`bridge_comparisons.prefix_ablation`).
- One trap: with the parent's straddle read slot (token space-merged with the turn's first word) the first-turn control fakes 0.14–0.32 via token leakage — the clean slot is the honest read.

### Result 4: Transfer of mapping from assistant to user
I then checked if the assistant mapping (in the chat template) transferred to the user (for each kind of user generated data)

![R² at each transfer tier, assistant-chat source map → user-chat targets per completion provenance, instruct, prefix arm (the construct-valid arm for user cells), L19, symlog y; dashed = each target's own ceiling](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/results_summaries/framing_character_user_turn/r4_assistant_to_user_ladder.png)

> Source: `eval_results/issue_1689/ladder/ladder_Qwen_Qwen2.5-7B-Instruct_L19.json` (parent round — so caveats (1)/(2) of Result 3 apply to the lmsys/on-policy series). Recapture-round direct transfer confirms the tier-1 read with corrected boundaries: assistant→user R² −3.67 (instruct) / −3.32 (base), barely above its shuffled-target null, while kNN retrieval stays 4–12× chance.

**Takeaways:**
- **The assistant map does not reconcile onto the user turn at any tier**: 0 of 9 assistant→user pairs reach the 90%-of-ceiling bar, both models (best recovery 0.43–0.88 instruct, 0.19–0.64 base).
- **The corrections that help are answer-side** — rotation (+0.41) and answer reparameterization (+0.60 of a 0.685 ceiling), while context-side rungs stay negative — the mirror image of the framing axis (Result 1), where context-side corrections do the work. Changing *who speaks next* changes where the map writes; changing *framing* changes how contexts are coordinatized.
- The reverse (user→assistant) "reconciles" everywhere, but only against tiny assistant-chat ceilings in that rig (0.253/0.077) — a low-ceiling artifact, not containment.

### Result 6: Differences between base and instruct model
Analysis of what remains the same vs different in base and instruct model (from all results above)

![Chat↔bare 9-tier ladder, base vs instruct, context arm, L19, n=4,724; dashed = each target's own ceiling](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/results_summaries/framing_character_user_turn/r6_base_vs_instruct_ladder.png)

**Takeaways:**
- **Same operator, different coordinates.** Within-framing ceilings are close (corrected: instruct 0.636 chat / 0.604 bare; base 0.517 / 0.549 — base's ordering flips, bare > chat), but direct transfer retains 83–84% of ceiling in instruct vs **15–31% in base**; both models reach ~ceiling at the context-reparameterization tier (base 0.561/0.530). Instruction tuning **canonicalizes the context coordinate system across framings rather than changing the operator**.
- Aligned operator similarity agrees: raw operator cosine 0.651 (instruct) vs 0.293 (base); Procrustes-aligned **0.855 vs 0.732** (rotation null ≈ 0.000) — both above the #825 base↔instruct cross-model anchor (0.686). So a framing change moves coordinates *less* than instruction tuning itself does (`eval_results/issue_1345/operator_comparison_{instruct,base}_context.json`).
- **Answer authorship reverses between models** (boundary-ablation round): instruct predicts chat ≥ bare ≫ story and pays more for its own answers than for framing (own-answer bare 0.384 vs embedded 0.628); **base is best in the story render with its own answers** (0.282–0.347 vs chat 0.179–0.207) — caveat: base story answers continue an instruct-written story prefix (`figures/issue_1345/story_boundary_ablation/tier_curves_provenance.meta.json`).
- **User turns behave identically in both models**: same provenance ordering, first-turn R² ≈ 0, assistant→user fails in both.
- **Are the axes related?** The framing axis and the model axis are the same *kind* of change (context-side coordinate moves over a shared operator, resolved at tiers 6–7); the speaker axis is a different kind (answer-side, unresolved at any tier); the story axis sits between — same operator partially recoverable, but with the target's own ceiling halved, so part of the story cost is not a coordinate change at all.

---

# Tiers of mapping transfer
LLM status:
- Wrote main ideas -> asked LLM to summarize -> lightly edited to de-slopify

*(Implementation note: the nine metrics below are exactly rungs 1–9 of the implemented ladder — direct / context offset / answer offset / bias refit / global scale / rotation / context reparam / answer reparam / full A·M·B.)*

# Metrics for mapping similarity

We refer to several different kinds of mapping similarity (between base and instruct model, from assistant to other characters in stories).

This note clarifies the different metrics for mapping similarity (and what each means on a mechanistic level). They are ordered from strongest to weakest alignment between the mappings (roughly — the exact containment relations are in the nesting note at the end).

**Setup:** `x` = context vector, `ŷ` = predicted answer vector; the source-setting map is `ŷ = W_s x + b_s`. Every metric keeps `W_s` frozen and fits only the stated correction in the target setting — more free parameters in the correction ⇒ a weaker claim about what is shared. `b*` denotes a bias refit in the target setting.

## Direct mapping transfer

- Train mapping `ŷ = W_s x + b_s` in source setting
- Apply the source map in target setting `ŷ_t = W_s x_t + b_s`
- Measures how much the context -> answer mechanism changes between the source and target setting

## Context offset

- Train mapping `ŷ = W_s x + b_s` in source setting
- Shift only the contexts in the target setting: `ŷ = W_s (x − Δx) + b_s`, with `Δx` = mean(target contexts) − mean(source contexts) (fit on the context clouds alone — never on context→answer pairs; needs no prompt pairing)
- The translation-only special case of "linear reparameterization of contexts" below (`A` restricted to a pure shift)
- **Interpretation:** the mechanism is untouched; the setting change adds one constant vector to every context representation, which does **not change the answers**. Making this context change ≡ steering with a vector at the context position: steer a target run by `−Δx` and the source map becomes exact

## Answer offset

- Train mapping `ŷ = W_s x + b_s` in source setting
- Shift only the answers in the target setting: `ŷ = W_s x + b_s + Δy`, with `Δy` = mean(target answers) − mean(source answers) (fit on the answer clouds alone)
- The translation-only special case of "linear reparameterization of answers" below (`B` restricted to a pure shift)
- **Interpretation:** the mechanism reads target contexts exactly as the source one would; the setting change is one constant vector pasted onto the answer after the map has run, independent of the question. Making this context change ≡ steering with a vector at the answer position

## Bias offset

- Train mapping `ŷ = W_s x + b_s` in source setting
- Refit only the bias in target setting: `ŷ = W_s x + b*` (`b*` fit by regression on target context→answer pairs)
- Contains both offsets above (`b* = b_s − W_s Δx`, `b* = b_s + Δy`, or any mix — equivalently, both translations at once): the constant correction is unconstrained, optimized on pairs, and makes no commitment about where it enters the computation
- Measures whether the **linear part** of the mechanism — which context directions move which answer directions, and by how much — is preserved, allowing an arbitrary constant shift

## Global scaling

- Train mapping `ŷ = W_s x + b_s` in source setting
- Fit a single scalar in target setting: `ŷ = α W_s x + b*`
- Measures whether the mechanism is preserved up to a uniform gain change: same read directions, same write directions, same relative strengths — only the overall magnitude of the context→answer effect changes (e.g. the whole map uniformly attenuated in the target setting)

## Mapping rotation

- Train mapping `ŷ = W_s x + b_s` in source setting
- Fit an orthogonal matrix in target setting: `ŷ = R W_s x + b*`, with `RᵀR = I` (orthogonal Procrustes)
- Which context directions the map reads, and how strongly, is unchanged; **where it writes** in answer space is rotated. Distances and angles among predicted answers are preserved
- Caveat: a singular-spectrum cosine is invariant to rotations on both sides and cannot establish this — only the fitted-`R` (direction-aware) read can

## Linear reparameterization of contexts

- Train mapping `ŷ = W_s x + b_s` in source setting
- Train linear mapping `A` from target contexts to source contexts (fit on contexts only)
- Apply the source map through it: `ŷ = W_s (A x) + b*`
- **Interpretation:** the mechanism and the answer coordinate system are shared; only the **coordinate system of the contexts** changes between settings

## Linear reparameterization of answers

- Train mapping `ŷ = W_s x + b_s` in source setting
- Train linear mapping `B` from target answers to source answers (fit on answers only)
- Apply `Bŷ = W_s x + b* => ŷ = B^-1 W_s x + b*`
- **Interpretation:** the mechanism and the context coordinate system are shared; only the **coordinate system of the answers** changes between settings

## Linear reparameterization of contexts and answers

- Train linear mapping `A` from target contexts to source contexts
- Train linear mapping `B` from target answers to source answers
- Apply same mapping `(Bŷ) = W_s (A x) + b* => ŷ = B^-1 W_s (A x) + b*`
- **Interpretation:** the input -> output relationship is preserved. The difference between the settings is the **coordinate system of the inputs and the outputs**
- You might think "we just showed that there is a new arbitrary linear mapping `ŷ = B^-1 W_s (A x) + b* = Mx + b*`" and this doesn't tell us anything
    - but the difference is that we are **never directly fitting our new learned mapping on the context -> answer mapping in the target setting**
    - we are showing something stronger:
        - the mapping from context to answer remains the same, but the context and answer representations change
