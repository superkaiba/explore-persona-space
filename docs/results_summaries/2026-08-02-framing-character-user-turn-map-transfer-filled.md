# How does the context→answer mapping transfer across framings, characters, and speakers? (filled)

*Filled 2026-08-02 from the banked artifacts of #1345 (framing transfer + story characters, incl. the #1887 λ-audit corrections), #1689 (21-condition speaker×framing lattice + the user-slot recapture round), and #825 (cross-model anchor). The draft's structure is kept verbatim; every number traces to the artifact named beside it. Terminology per `docs/glossary_context_answer_map.md`: **context vector $v_C$** = activation at the last prompt token, **answer vector $v_A$** = mean (or end-token, stated per read) activation over the answer, **context map** $M'$: $v_A \approx M' v_C$. Where a requested cell was never run, that is stated instead of approximated.*

## Motivation
- We've been studying this mapping from context to answer
- I wanted to see:
    - how this mapping transfers from chat template to no chat template to realistic story framing (with assistant, user, fictional characters)
        - and how this changes from base to instruct model
        - and the relationship between chat template to no chat template to realistic story framing for each character
        - and the relationship between each character for each framing
        - and if those axes are in any way related

## Methodology

**Two experiments carry these results.** #1345: assistant-only framing transfer on 4,724 matched single-turn LMSYS conversations (the full 9-rung ladder ran on chat ↔ bare text; story was measured at rung 1 and rung 9 only). #1689: a 21-condition speaker×framing lattice (7 identities × 3 framings) on 3,800 two-turn LMSYS conversations, all 126 ordered pairs through the same 9-rung ladder, plus a `user-slot-recapture` follow-up round that fixed the user-cell read boundaries. Both: Qwen2.5-7B base AND Instruct, teacher-forced capture at layer 19 of 28, closed-form ridge with conversation-grouped 5-fold CV, shuffled-Y nulls, identity+learned-bias baseline and kNN retrieval on every fitted map.

- I considered **3 framings** (regime codes r1/r2/r3-r4 in the artifacts). The same exchange transposed into each (story render verbatim from `data/issue_1345/story_boundary_ablation/stories/kept_stories_bnd_v4_instruct.jsonl`, conv s2541; render formats `scripts/issue825_render_formats.py:179-233`, `scripts/issue1345_common.py:592-605`):
    - **Chat template** — `<|im_start|>user\nWhat's the time now?<|im_end|>\n<|im_start|>assistant\n18:45, or 6:45 PM, according to your device.<|im_end|>` (raw special-token render, NOT `apply_chat_template` — no default system preamble). Context read slot: last token of the assistant header.
    - **Bare text (no template)** — `User: What's the time now?\n\nAssistant: 18:45, or 6:45 PM, according to your device.` Read slot: last token of `Assistant: `.
    - **Story (narrative prose)** — *"The sun had just set, casting a golden glow over the quiet park. Mia stood by a bench... 'What's the time now?' she asked, tapping her phone impatiently.\n\nAssistant replied: "18:45, or 6:45 PM, according to your device.""* Read slot: the colon of the attribution `Assistant replied:`. (Character name was `ARIA` in the first round, renamed to `Assistant` for all follow-up rounds; the rename alone dropped story yield 84.0%→72.2%.)
- I considered **4 story characters** plus the assistant and the user. #1345's panel (verbatim descriptions, `scripts/issue1310_common.py:128-134`), ordered by judge-scored AI-likeness of their own answers (0–100, Sonnet-4.5 judge, 5 draws/item, 300 items/cell; base-model values, `figures/issue_1345/story_boundary_ablation/character_ai_likeness.meta.json`):
    - **HELIOS** — "a calm, precise artificial intelligence" — most AI-like (74.2)
    - **Wren** — "a warm, endlessly helpful assistant who patiently helps anyone who asks" (63.5)
    - **Vex** — "a theatrical, scheming villain who delights in menace" (57.5)
    - **Dana** — "an ordinary, unremarkable everyday person" — least AI-like (53.2)
    - #1689's lattice uses HELIOS ("an AI character"), Wren ("an assistant-like human"), Dana ("an ordinary person") — no Vex.
    - *Same-context per-character response examples were generated (16 cells) but the kept stories live only on HF (`issue1345_framing` raw-completions prefixes) — not pulled here; only the judge legs landed in `eval_results/issue_1345/judge_legs/`.*
- For the story framing I ran **2 data conditions** (#1345, instruct only — base story generation missed its yield floor, 96/500 kept, and was halted: `eval_results/issue_1345/story_regime_coverage.json`):
    - **inserted** — answer generated on-policy in chat, then embedded verbatim into the story render (kept 2,164/2,700; dominant reject: the model paraphrases instead of embedding)
    - **generated in target** — the model writes its own answer inside the story (kept 2,019/3,438 = 58.7%)
- Everything above ran in **both base and instruct**, except base story cells (yield-halted; base own-answer story cells exist only in the boundary-ablation round).
- **Estimator correction that changes older numbers:** the ambient-basis story fits were under-determined (n_train 1,614–1,730 < d = 3,584) and GCV produced spuriously negative R²; the #1887 audit replayed all 67 cells (gate PASS, |ΔR²| ≤ 0.001) and the **corrected reduced-basis (train-fold PCA) column is the headline everywhere below** (`eval_results/issue_1345/lambda_audit_1887/corrections_table.md`). Figures rendered before 2026-07-30 still carry the uncorrected values — noted per figure.

I considered the tiers of mapping transfer at the bottom of this page — the 9 tiers there are exactly the implemented rungs 1–9 (`scripts/issue1345_ladder_rungs.py`; #1689 uses the same rung set).

I considered the held-out $R^2$ after applying each of these transformations on a fixed training set of generic prompts (LMSYS-derived; corrections always fit on the target train fold, source operator frozen).

The goal is to see how transferrable the mapping is between different settings

## Results
I first considered the instruct model.
### Result 1: Effect of changing **framing** on the assistant mapping

I wanted to see the effect of changing the **framing** of the conversation on the assistant mapping, so I plotted the $R^2$ at each transfer tier for:
- assistant chat template -> assistant bare text
- assistant chat template -> assistant in story
- for both on-policy generated and inserted text

**Coverage note (honest):** the 9-rung ladder ran only on chat ↔ bare text (inserted/teacher-forced text, n=4,724). The story arm was measured at rung 1 (direct) and rung 9 (full reparameterization) only, in both data conditions. *This ladder round was never folded into #1345's body — these numbers are surfaced here for the first time* (`eval_results/issue_1345/ladder_rungs/ladder_rungs_instruct_context.json`, commit `ec14b07010`).

![9-rung transfer ladder, chat <-> bare text, context arm, L19, both models; corrections fit on target train fold, source operator frozen](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1345/ladder_rungs/ladder_hero_context.png)

Instruct, context arm ($v_C \to v_A$), held-out $R^2$ (target's own ceiling in parens):

- chat→bare (ceiling 0.625): direct **0.520** · ctx offset 0.554 · ans offset 0.517 · bias refit 0.559 · scale 0.554 · rotation 0.592 · **ctx reparam 0.606** · ans reparam 0.520 · full A·M·B **0.620**
- bare→chat (ceiling 0.654): direct **0.552** · ctx reparam **0.638** · full **0.654** (other rungs same pattern)
- Shuffled nulls −0.03 to −0.08 all rungs except rotation (+0.12–0.14, null inflated by construction). kNN retrieval fold-0 (n_pool 945, chance@1 0.001): ceiling acc@1 0.671, direct 0.571, full 0.668.

Chat→story has no ladder; the two measured tiers (`eval_results/issue_1345/cross_regime_transfer_instruct_context.json`, `.../conversation_paired_stories_assistant/reparam_recovery_r1_r4_instruct_context.json`, `.../onpolicy_assistant_story/reparam_recovery_r1_r4_instruct_context.json`):

- **Direct transfer collapses in every story direction** (ambient basis: chat→story −3.06, story→chat −7.02; vs chat→bare +0.52).
- **Full reparameterization is strikingly one-way.** Story-operator-into-chat recovers **0.610** (inserted) / **0.564** (on-policy) — essentially the corrected matched-row chat ceiling (0.609 / 0.567). Chat-operator-into-story fails: **−0.17** (inserted) / **+0.16** vs corrected story ceiling 0.26 (on-policy).
- Within-framing corrected $R^2$, matched rows, same folds: **inserted — story 0.367 vs chat 0.609; generated-in-story — story 0.262 vs chat 0.567** (`lambda_audit_1887/corrections_table.md`). Estimator-free corroboration: rank-1 retrieval story-context→paired-chat-context 354/2,163 vs chance ~1 (`story_context_info_probe/summary.json`); a 512-unit MLP does not beat ridge, so this is not linear-estimator failure.

![Within-framing R² on matched rows, inserted condition; bars predate the #1887 correction — corrected values: story 0.367, chat 0.609, bare 0.574](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1345/conversation_paired_stories_assistant/framing_effect_matched_rows_bars.png)

**Takeaways:**
- Chat and bare text share **one operator up to linear coordinate changes**: context-side reparameterization alone recovers 97–98% of ceiling; answer-side reparameterization does nothing (rung 8 ≈ rung 1). The coordinate change lives on the **context side**.
- Story framing **weakens the map ~2× but does not eliminate it** (0.367 vs 0.609 inserted; 0.262 vs 0.567 on-policy); generating the answer in-story costs a further ~0.10 beyond the framing effect.
- The story deficit is a **context-coordinate degradation, not an answer-side change**: story answer vectors remain predictable from a chat-trained operator after context realignment (0.61 ≈ chat ceiling), while chat operators cannot be carried into story coordinates. Read-slot choice doesn't explain it (slot ablation: best slot 0.453 vs chat 0.609, deficit CI wholly below zero).

### Result 2: Transfer of mapping between characters
I then wanted to see if the assistant was a privileged character/persona when it comes to predicting answer activations from context, so I plotted the $R^2$ at each transfer tier for the assistant in story mapping to the other character in story mapping, for both on-policy generated and inserted text

**What actually exists (honest):** per-character maps were never fit in the strong #1345 rig — all 16 character capture cells are banked (`data/issue_1345/char_*/turnstore/`) but only the judge leg landed. The character-pair ladder exists only in #1689, **where every story/character cell's own within-cell ceiling is ≈ 0** (instruct: assistant_story 0.016, helios_story 0.001, dana_chat 0.219; `eval_results/issue_1689/ladder/ladder_Qwen_Qwen2.5-7B-Instruct_L19.json`), so those transfer reads are low-ceiling artifacts, not measurements. #1689's promoted headline — *changing surface framing costs the map far more than changing speaker identity* (LOW confidence) — rests on this pattern: same-framing identity swaps reconcile at rung 1 (e.g. assistant_chat→wren_chat direct 0.278 vs ceiling 0.127), while framing swaps need rung 9 or never reconcile.

![Full 126-pair rung lattice, instruct, both arms; speaker-identity pairs reconcile at low rungs, framing pairs at rung 9 or never — read with the near-zero story-cell ceilings in mind](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1689/fig1_rung_heatmap_instruct.png)

What the character panel does support — a judge-scored AI-likeness axis over the characters' own story answers (0–100, Sonnet-4.5, 5 draws/item, 300 items/cell, Batch API; `eval_results/issue_1345/judge_legs/judge_legs_summary.json`):

![Judge-scored AI-likeness of each character's own story answers vs the injected verbatim-reference control, base model; ±1.96 SE ≈ 1.7](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1345/story_boundary_ablation/character_ai_likeness.png)

**Takeaways:**
- Own-answer AI-likeness orders **HELIOS 74.2 > Wren 63.5 > Vex 57.5 > Dana 53.2** (base; instruct 72.1/61.8/55.4/53.1) — but the **injected control (same reference answer under every character) is flat at 78.7–80.8**, so the axis reads *who wrote the answer*, not which character is speaking.
- **"Is the assistant privileged among characters?" is still unanswered at the map level.** The needed fits are ~0 GPU-h on banked captures (#1345 character turnstores); caveats if run from #1689 instead: near-zero ceilings, 5/8 reference-answer character cells missed generation-yield floors.
- Weak provenance-side hint from the #1689 recapture round: authoring the assistant's reply as Wren instead of the assistant changes second-user-turn predictability by ≤0.03 $R^2$ (0.356 vs 0.331 base; 0.316 vs 0.304 instruct).

### Result 3: User turn mapping
I then wanted to see if the user character was just another similar character or in some way privileged. For the user turn I considered 3 types of completions:
- Haiku generated
- real user data
- on-policy generated by Qwen

**Three provenance caveats first (they change every read):**
1. **The "real user data" arm is not real user text.** The two-turn LMSYS corpus has no u2 field; `scripts/issue1689_render_conditions.py:309` falls back to the constant string "Can you say a bit more about that?" on every row (verified: one distinct sha256 across 2,114 rendered rows). The prefix (u1, a1) is real LMSYS; the predicted turn is a constant — which is why that column is near-perfectly predictable. **#1689's promoted body still calls this arm "tier-1 real data"; that claim is refuted and awaits the deferred body-fold correction.**
2. In the parent round the on-policy arm was a **byte-duplicate of the Haiku arm at/before capture** (identical ladder outputs to machine precision; raw text differs on 3,800/3,800 rows). Only the recapture round gives a genuine third arm.
3. **Qwen largely refuses to play the user**: all six on-policy cells missed the 80% yield floor (chat yield 2.5% instruct / 4.2% base — n=266/473 rows; `eval_results/issue_1689/onpolicy_stats/`). On-policy chat cells are partly a sample-size story.

Recapture-round held-out $R^2$, second user turn, clean read slot (X = last token before the turn, Y = turn end token; HF mirror `issue1689_speaker_lattice/user_slot_recapture/eval_mirror/user_slot_recapture/summary.json` — **not yet committed to git, no figures rendered; the readout round is deferred**):

| framing | model | const-"lmsys" | Haiku | on-policy Qwen |
|---|---|---|---|---|
| chat | Instruct | 0.693 | 0.243 | 0.069 |
| chat | base | 0.629 | 0.265 | 0.006 |
| bare text | Instruct | 0.701 | 0.316 | 0.018 |
| bare text | base | 0.762 | 0.344 | 0.021 |
| story | Instruct | 0.733 | 0.236 | 0.008 |
| story | base | 0.769 | 0.244 | 0.005 |

**Takeaways:**
- $R^2$ is **monotone in how constrained the turn is**: constant string ≈ 0.63–0.77, Haiku-simulated ≈ 0.24–0.34, the model's own free generation ≈ 0. No genuine real-user-text arm exists yet.
- The **target convention matters a lot**: switching Y to the mean over the turn lifts e.g. chat/Instruct/Haiku 0.243→0.374 and bare/Instruct/on-policy 0.018→0.287 — much of the apparent unpredictability is the end-token read, not content.
- Framing barely matters for the user turn (story ≈ chat ≈ bare per provenance), and the story character label (Alex vs literal "User") is nearly free (Δ ≤ 0.01).

I then looked at if this $R^2$ changed if you ran this at the second user turn (intuitively the first user turn is unpredictable because the model has no prior information about the user)

Numbers from the same recapture `grid_r2` (no figure yet): **first user turn $R^2$ = −0.0016 to −0.075 in every cell, both models, all framings — exactly zero**; second user turn rises to the table above (e.g. Instruct/chat/Haiku, Y_mean: u1 −0.001 vs u2 0.374). One trap: with the parent's straddle read slot (token space-merged with the turn's first word) the u1 floor control fakes 0.14–0.32 via token leakage — the clean slot is the honest read.

**Takeaways:**
- **The hypothesis holds cleanly**: with no prior user content the first user turn is unpredictable (R² ≈ 0); one (u1, a1) exchange of context buys ~0.24–0.37 for a simulated user. Dropping (u1, a1) from the second-turn read costs 0.04–0.09 directly.

### Result 4: Transfer of mapping from assistant to user
I then checked if the assistant mapping (in the chat template) transferred to the user (for each kind of user generated data)

Recapture direct transfer (no corrections, `cross_role_transfer`): every direction fails hard — assistant→user $R^2$ **−3.67** (Instruct) / **−3.32** (base), user→assistant −2.18 / −1.10, each only marginally beating its shuffled-target null; yet kNN retrieval stays 4–12× chance (the recurring R²-vs-retrieval dissociation).

Parent 9-rung ladder, assistant_chat → user cells (worked case Instruct → user_haiku_chat, ceiling 0.685, bar 0.616): direct **−4.20**, ctx offset −4.20, ans offset −0.29, bias refit −0.29, scale −0.15, **rotation +0.41**, ctx reparam −0.28, **ans reparam +0.603**, full +0.603 — best recovery 0.88 of ceiling, **never reaches the bar**. Across all assistant→user pairs: **0 of 9 reconcile at any rung, both models** (best-rung recovery 0.43–0.88 instruct, 0.19–0.64 base). The reverse direction (user→assistant) "reconciles" everywhere, but only because the assistant-chat target ceiling there is 0.253/0.077 — a low-ceiling artifact, not containment. (`eval_results/issue_1689/ladder/ladder_*_L19.json`)

![Per-rung recovery for user-provenance pairs, both models; parent round — read with caveats 1–2 of Result 3](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1689/fig6_provenance_ladder.png)

**Takeaways:**
- The assistant map does **not** transfer to the user turn at any tier — and unlike the framing axis (context-side fix, Result 1), the partial recovery that does exist comes **entirely from answer-side corrections** (rotation + answer reparameterization; context-side rungs stay negative). Changing *who speaks next* changes where the map writes; changing *framing* changes how contexts are coordinatized.
- Provenance ordering under transfer matches Result 3: Haiku/const chat targets recover best (0.88), on-policy and bare-text targets worst (0.43–0.68).

### Result 6: Differences between base and instruct model
Analysis of what remains the same vs different in base and instruct model (from all results above)

- **Within-framing ceilings are close** (corrected: instruct chat 0.636 / bare 0.604; base chat 0.517 / bare 0.549) — but the ordering flips: base predicts bare text *better* than chat.
- **Direct transfer is where they diverge**: chat↔bare direct retains 83–84% of ceiling in instruct vs **15–31% in base** (base chat→bare 0.177/0.578, bare→chat 0.082/0.542). Both models reach ~ceiling at rung 7 (context reparam: base 0.561/0.530) — same operator, different coordinates.
- **Aligned operator cosine says the same thing**: raw 0.651 (instruct) vs 0.293 (base); Procrustes-aligned **0.855 vs 0.732** (rotation null ≈ 0.000) — both above the #825 base↔instruct cross-model anchor (0.686). Instruction tuning **canonicalizes the context coordinate system across framings rather than changing the operator**. (`eval_results/issue_1345/operator_comparison_{instruct,base}_context.json`)
- **Answer authorship reverses between models** (boundary-ablation round, own-answer cells): instruct predicts chat ≥ bare ≫ story and pays more for its own answers than for the framing (own-answer bare 0.384 vs embedded 0.628); **base is best in the story render with its own answers** (0.282–0.347 vs chat 0.179–0.207). Caveat: base story answers continue an instruct-written story prefix. (`figures/issue_1345/story_boundary_ablation/tier_curves_provenance.meta.json`)
- **User turns behave the same in both models**: identical provenance ordering (const > Haiku > on-policy), first-turn R² ≈ 0 in both, assistant→user fails in both.

![Within-cell R² across read positions per framing × answer provenance, base vs instruct; the base own-answer ordering reversal](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1345/story_boundary_ablation/tier_curves_provenance.png)

**Takeaways (are the axes related?):**
- The **framing axis and the model axis resolve at the same tier** — both are context-coordinate changes over a shared operator (rung 7 recovers both; aligned cosines 0.855/0.732 within-model across framings vs 0.686 across models within-framing, i.e. a framing change moves coordinates *less* than instruction tuning does).
- The **speaker axis is a different kind of change** — answer-side (Result 4), not context-side, and not reconciled by any tier tried.
- The **story axis** sits in between: same operator recoverable one-way (story answers still live in reachable coordinates) but the story context coordinates are degraded, ~2× R² cost, in both models' own-answer regimes it's the *base* model that is most story-native.

## What was NOT run (so this page doesn't overclaim)
1. **Per-character transfer maps** (the literal Result-2 ask): captures banked in `data/issue_1345/char_*/turnstore/`, fits ≈ 0 GPU-h. The only existing character-pair ladder (#1689) is ceiling-limited.
2. **A genuine real-user u2 arm** (needs a corpus with real second turns), and the #1689 recapture readout round: commit the HF-mirrored results, render figures, fold into the body, and correct the body's "tier-1 real data" claim.
3. **Base story cells** in the main rounds (yield-halted at 96/500) and any story-arm run of the full 9-rung ladder (currently rungs 1+9 only, and at reduced basis never).

---

# Tiers of mapping transfer
LLM status:
- Wrote main ideas -> asked LLM to summarize -> lightly edited to de-slopify

*(Implementation note: the nine metrics below are exactly rungs 1–9 of the implemented ladder — direct / context offset / answer offset / bias refit / global scale / rotation / context reparam / answer reparam / full A·M·B: `scripts/issue1345_ladder_rungs.py`, `eval_results/issue_1689/ladder/`.)*

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
