---
title: The context→answer mapping is inherited from pretraining
kind: experiment
tags:
- keep-running
- followup-manual
created_at: '2026-07-02T00:14:16Z'
has_clean_result: true
parent_id: 779
origin_prompt: 'Help me to plan this experiment: Is the context vector to answer profile
  mapping present in the base model (Qwen/Qwen2.5-7B) and does it hold for the user?
  (verbatim full prompt in ## Provenance; follow-ups: frame on #779''s per-context
  map, Haiku-4.5-as-user generated conversations, file as proposed only)'
goal: 'Test whether the per-example linear context-to-answer-profile map h: c_x ->
  v(x) (#779 recipe: held-out K-fold ridge over thousands of per-example pairs) exists
  in the pretrained Qwen/Qwen2.5-7B vs Qwen2.5-7B-Instruct, for both the assistant
  and the user turn, under chat-template vs naturalistic formatting; secondary: per-position
  decay and cross-role prediction beyond a topic-persistence baseline.'
relates_to:
- identity-contextual-vs-base
- identity-cb-duality
---
# Result: The context→answer mapping is inherited from pretraining

## Motivation

* An earlier experiment found a linear mapping from a single context vector to the mean answer vector in the instruct model (R² ~0.7, [#779](https://eps.superkaiba.com/tasks/779))
* I wanted to characterize where that mapping comes from and what carries it:
    * does it exist in the pretrained-only model, and what does post-training change?
    * does it require the chat-template tokens?
    * is it a general character→dialogue mechanism (fiction), or generic next-span prediction, or something chat-specific?
    * which part of the input carries it, and how far ahead does it reach?

## TLDR

- **The mapping already exists almost entirely in the pretrained base model:**
    - held-out R² 0.588 for base vs 0.673 for instruct at the shared best layer **~87% of instruct prediction strength**
- **Post-training reparameterizes the existing mapping by a general linear map**
    - the base map, run through a fitted general-linear change of coordinates, has as much predictive power as the instruct map on instruct model text
- **The mapping holds up without chat-template tokens (in both base and instruct model):**
    - refit on the same single-turn conversations rendered as plain "User:/Assistant:" text: instruct $R^2 = 0.625$ (vs $0.654$ chat), base $R^2 = 0.578$ (vs $0.542$ chat) at layer 19
    - small format×model crossover, both sides distinguishable from zero under a paired bootstrap: removing the template costs instruct ~0.03 R² and gains the base ~0.04
    - without the template the base/instruct strength ratio rises to ~93% (from ~83% chat at matched n)
- **The mapping does not hold up for the user turn:**
    - the user's next turn is linearly unpredictable (ridge R² negative for both real human turns and model-generated user turns) and only weakly nonlinearly predictable (MLP 0.19–0.23)
- **The mapping does NOT hold up for generic stories (off-policy or on-policy):**
    - re-training the mapping on generic stories gets $R^2 \approx 0.16$ for both on-policy generated stories and off-policy stories taken from the internet
- **The mapping is not just generic next-span prediction:**
    - training a mapping from punctuation tokens to the span of text before the next punctuation token in generic text gives $R^2  \approx 0.05/0.1$ in the base/instruct models
- **When predicting the assistant's response in two turns, the linear mapping fails, but a nonlinear mapping survives:**
    - ridge regression $R^2 \approx −0.46$ to $-0.08$ vs MLP $R^2 \approx 0.49–0.56$
- **In real multi-turn conversations the per-turn mapping runs at roughly half the single-turn strength, and answer provenance explains only part of the gap:**
    - regenerating each answer on-policy lifts turn 1 from $R^2 = 0.21$ to $0.35$ (vs the $0.48$ single-turn anchor at the same n) and the pooled all-turn read by only +0.04 (instruct) — most of the multi-turn drop is the real-user corpus, not who wrote the answer
    - in the pretrained model on-policy answers are slightly *less* predictable (pooled −0.02)

## Methodology

- **Models:** Qwen2.5-7B (pretrained base) vs Qwen2.5-7B-Instruct
- $v_C$: activation at the end of the context (taken at best layer from prior experiments)
- $v_A$: the mean activation over the answer span (taken at best layer from prior experiments)
- **Estimator:** ridge regression for linear, MLP for nonlinear
- **Metric:** held-out R² (variance-weighted over the 3,584 dims

## Results:

### _Result 1: The mapping already exists almost entirely in the pretrained base model_

I first wanted to see if the mapping already exists in the base model

**Methodology (this result):**
- 5000 LMSYS user turns
- answers generated either by base model or instruct model
- 5-fold cross validation across 4000 samples (chat templated data in both base **and** instruct model)
- compute mapping in base model
- compute mapping in instruct model
- Compare $R^2$ for both at all layers
- Baseline = shuffled context/answer pairings

![Held-out R² by layer for the instruct and pretrained-base context→answer maps, with the shuffled-pairing null near zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/43d7a7e3ea5b77cb540c31e182b4eb02d52f801f/figures/issue_825/s_track_layer_curves.png)

**Takeaways:**

- The mapping already mostly exists in the base model:
    * Base **0.588** vs instruct **0.673** at the shared best layer 19 — **87.3%** of instruct strength

### _Result 2: Post-training reparameterizes the existing map by a general linear map_

I then wanted to see whether post-training builds a new map or just re-expresses the existing one

**Methodology**
- Fit the ridge operator M: context --> answer separately in base and instruct
- Fit a change-of-coordinates on the context side and answer side between the 2 models on identical text
- Test whether the base operator, reparameterized, is as good of a predictor of instruct text as the instruct operator

![Held-out R² predicting instruct answers: instruct own map 0.673, base own map 0.588, base map reparameterized by a general linear change of coordinates 0.673](https://raw.githubusercontent.com/superkaiba/explore-persona-space/43d7a7e3ea5b77cb540c31e182b4eb02d52f801f/figures/issue_825/reparam_vs_instruct.png)

**Takeaways:**
- Reparameterizing the base model mapping with a general linear mapping gives as good of a predictor of instruct text as the instruct mapping
- This indicates the information to predict the instruct text is already present in the base model context vector
    - this gives some indication that post-training is only eliciting capabilities and not teaching new ones (at least the ones that are linearly predictable from the context vector)

### _Result 2.5: What is the reparameterized base map?_

I then wanted to see if there is a difference between the reparameterized base map and the instruct map (just because they have the same predictive power doesn't mean they are the same)

**Takeaways**
- Yes there is:
    - if you rotate them to be as similar as possible then flatten and take cosine similarity, you only get 0.69
    - so you need rescaling of directions to turn the base map's into the instruct's one
- Where is the change happening?
    - how well instruct's context vectors can be linearly reconstructed from base's: $R^2=0.619$
    - how well instruct's answer vectors can be linearly reconstructed from base's: $R^2=0.902$
    - so most of the reparameterization is on the context side

### _Result 3: The mapping holds up without chat-template tokens (in both base and instruct model)_

I then wanted to see if this mapping was only there because of the chat-template tokens.

**Methodology:**
- Re-render the same Track-S conversations (same response texts) with the chat template replaced by plain "User:" + "Assistant:" text, then refit the same ridge recipe in both models
    - [Examples](https://htmlpreview.github.io/?https://raw.githubusercontent.com/superkaiba/explore-persona-space/c1fe5a3ee6a8cd96b19f854d0e27a69e2be18e34/experiments/dashboards/issue825_naturalistic_examples.html)
- 4,724 of the 5,000 conversations survive the render/span-alignment filter; the paired contrast refits BOTH formats on this shared set (identical folds, seed 0) with a 1,000-draw conversation-level bootstrap on the naturalistic−chat delta
- Anchor gate: the chat refit first reproduces the committed Result-1 values (instruct 0.673 / base 0.588 at layer 19) to within 0.001

![Held-out R² by layer, chat template vs naturalistic User:/Assistant: render, per model, both refit on the shared 4,724 conversations](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c3a01240f0c50a889be24075b1fa48168e20b240/figures/issue_825/nat_s_layer_curves.png)

![Paired naturalistic minus chat R² delta per frozen layer per model, with the conversation-level bootstrap interval](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c3a01240f0c50a889be24075b1fa48168e20b240/figures/issue_825/nat_s_l19_delta.png)

**Takeaways:**
- The mapping **holds up** without chat-template tokens in both models: instruct $R^2 = 0.625$ (vs $0.654$ chat), base $R^2 = 0.578$ (vs $0.542$ chat) at layer 19 — all four cells sit ~0.6 above the shuffled-pairing null (≈ −0.03)
- Small format × model crossover, both deltas distinguishable from zero under the paired bootstrap: the template helps instruct slightly (−0.029 R² without it) and hurts the base slightly (+0.037 without it); the base advantage holds at all four frozen layers, the instruct cost is concentrated at layers 18–19
- Without the template the base reaches **~93%** of instruct strength (vs ~83% chat at matched n) — the pretraining-inheritance picture gets stronger, not weaker, off-template
- Per-layer view: [naturalistic vs chat R² scatter, one point per layer](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c3a01240f0c50a889be24075b1fa48168e20b240/figures/issue_825/nat_s_format_scatter.png) — instruct layers sit below the identity line, base layers above
- This indicates it is not a function of the chat template but more a function of general assistant -> output text
- Caveats: an earlier draft of this section quoted R² 0.71–0.74, which came from the [#1092](https://eps.superkaiba.com/tasks/1092) naturalistic-transcript corpus — the numbers above are the final refit on this experiment's own conversations; the result JSONs were recovered from the crashed run's boot disk (the run finished fitting and died uploading); the planned MLP probe on the naturalistic cells was not run (ridge-only round)

### _Result 4: The mapping does NOT hold up for generic stories, or for the user turn, or for generic next span prediction_

I then wanted to see
- is this just a generic "character -> character output" mapping in stories(as might be predicted by PSM)
- does this mapping hold for the user turn?
- does this mapping hold for generic next span prediction (punctuation to span of text before next punctuation)

**Methodology:**
- Fit mapping to:
    - character -> answer in generic real stories
    - character -> on-policy generated answer in stories generated by the model
    - user context -> user answer (from real data)
    - user context -> user answer (generated by model)
    - punctuation token -> next span of text before next punctuation (real generic data)
    - punctuation token -> next span of text before next punctuation (generated by model)
- Plot $R^2$ for each

![Within-regime held-out R² (ridge, layer 19) for chat reference, real/model stories, real/model-generated user turns, and real/model-generated next-span cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/43d7a7e3ea5b77cb540c31e182b4eb02d52f801f/figures/summaries/context_answer_map/within_regime_r2.png)

**Takeaways:**
* The mapping does not hold up for:
    * generic stories (on and off policy)
    * the user turn
    * generic next-span prediction:
- I am thinking this is probably because it might be a "per-character" mapping (although the user turn not holding up somewhat disproves this)?
    - potentially it is an "AI character mapping" because the model thinks it should be good at predicting text generating by an AI

### _Result 4.5: Story characters DO carry the map once properly powered — the initial weak read was an n≪p artifact (instruct control pending)_

I wanted to test this hypothesis and see if a mapping trained on single character outputs in stories existed.

**Methodology**
- Tell model to generate stories with 4 different characters:
    - Wren: warm assistant
    - HELIOS: calm AI
    - Dana: ordinary person
    - Vex: theatrical villain
    - [Examples](https://htmlpreview.github.io/?https://raw.githubusercontent.com/superkaiba/explore-persona-space/c1fe5a3ee6a8cd96b19f854d0e27a69e2be18e34/experiments/dashboards/issue1310_story_examples.html)
- Both base and instruct model
- Plot $R^2$ for each character

![Per-character held-out R² (layer 19) for Wren, HELIOS, Dana, Vex in base and instruct, against the chat assistant ceilings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/43d7a7e3ea5b77cb540c31e182b4eb02d52f801f/figures/issue_1310/perpersona_r2_clean.png)

**Takeaways**
- The weak/inconsistent read in the figure above was an underpowered v2 artifact (n ≪ p; e.g. the instruct Vex / swap cells sat at n = 149 / 524); the [#1310](https://eps.superkaiba.com/tasks/1310) v3 re-run (prefill on-policy story datagen, ~1.3–3.1K pairs per character) flips it
- v3 base model held-out $R^2$: Wren 0.137 / HELIOS 0.148 / Dana 0.147 / Vex 0.106; correct-character pairing 0.232 vs cross-character swap −0.004 → the map is **per-character, not generic** (which also explains Result 4's pooled generic-story null: a cross-character map is ≈ 0)
- v3 instruct (partial — the run crashed before instruct Vex + the instruct swap control): Wren 0.235 / HELIOS 0.253 / Dana 0.188 — positive and stronger than base
- This largely dissolves the "why does User:/Assistant: hold up when story-like doesn't" question: story characters DO carry the map — the assistant's is just ~3–5× stronger (~0.65 chat) — so the open question is the **magnitude gap**, not presence/absence
- HELIOS (the AI character) ≈ Wren/Dana cuts against the "AI character mapping" hypothesis from Result 4 — being an AI character alone doesn't buy assistant-level strength
- Caveats: the figure still shows the superseded v2 read (v3 figures pending); v3 numbers are from the #1310 v3 run marker with data durable in the HF crash-persist (`issue1310_partial/att-20260715-052017`); the instruct specificity control (instruct Vex + swap) is incomplete — instruct-tail re-run dispatched 2026-07-16

### _Result 5: The linear mapping breaks across two-turns, but the nonlinear mapping holds_

I then wanted to see if you can predict the assistant answer in 2 turns from the current turn's context (both linearly and nonlinearly)

**Methodology:**
- Train linear and nonlinear mapping from context vector to answer vector for assistant **in 2 turns**
- Plot $R^2$

![Held-out R² at layer 19 for ridge vs MLP: single-turn reference cells vs predicting the assistant answer two turns ahead](https://raw.githubusercontent.com/superkaiba/explore-persona-space/43d7a7e3ea5b77cb540c31e182b4eb02d52f801f/figures/issue_825/ridge_vs_mlp_2turn.png)

**Takeaways:**
- There is no linear mapping from context vector to answer vector for the assistant in 2 turns
- There **is** a nonlinear mapping from context vector to answer vector for assistant in 2 turns ($R^2 \approx 0.49-0.56$)
- This suggests the information is there but it is not linearly decodable - kind of a strange result... where does the nonlinearity come in?

### _Result 6: The per-turn mapping persists at every turn depth at roughly half single-turn strength — on-policy answers recover only part of the gap_

The single-turn mapping ($R^2 = 0.48$ at matched n) predicts the model's OWN answers, but on real logged multi-turn conversations the per-turn map came out much weaker. I wanted to isolate whether that residual comes from answer provenance (the logged answers were written by other models/humans) or from the multi-turn real-user corpus itself.

**Methodology:**
- 497 real logged multi-turn conversations (WildChat/LMSYS, ≥2 turns; 2,572 turn–answer pairs; per-turn cells kept at n ≥ 30, assistant turns 1–23)
- Contexts identical between arms: the real logged conversation up to each turn-k user query
- Two answer arms per turn: the real **logged** assistant answer vs the model's **own** on-policy answer to the same query (one draw, temperature 1.0, matching the single-turn generation recipe)
- Same ridge recipe (held-out $R^2$, grouped 6-fold CV), layer-19 headline; shuffled-answer null per turn; paired conversation-level bootstrap on the own−logged delta; both mapping arms fit — context (everything before the answer, including the query) and prefix (everything before the user query)
- Single-turn anchor re-verified this round: $R^2 = 0.476$ at matched n=497

![Per-turn held-out R2 at layer 19 for own on-policy vs logged answers, instruct and pretrained panels, with the shuffled null band and the single-turn anchor line at 0.476](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b610115c89304bebecd7d52b718fa335bb53f9b7/figures/issue_825/onpolicy_turn_depth.png)

**Takeaways:**
- The mapping exists at **every** turn depth — all cells clear the shuffled null by a wide margin — but at roughly half the single-turn strength, decaying slowly with depth (instruct own: 0.35 at turn 1 → ~0.11 by turn 19)
- On-policy answers help most at turn 1 (instruct: $R^2$ 0.212 → 0.346) and only +0.04 pooled across all turns, both distinguishable from zero under the paired bootstrap; the two curves converge by turn ~13
- Even with own answers, turn 1 reads 0.35 vs the 0.476 single-turn anchor at the same n — provenance contributes, but **most of the multi-turn gap is the corpus** (real-user multi-turn conversations vs the curated single-turn set), not who wrote the answer
- In the pretrained model own answers are slightly *less* predictable (pooled −0.02; turn 1 −0.08): the base model's temperature-1.0 completions on the raw-text multi-turn render are noisier than the logged ones
- The prefix arm stays near zero at every depth (−0.10 to +0.08, both models, both provenance arms; structurally degenerate at turn 1) — the map runs through the query-inclusive context, not the pre-query state alone ([prefix-arm figure](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b610115c89304bebecd7d52b718fa335bb53f9b7/figures/issue_825/onpolicy_turn_depth_prefix.png))
- Not an answer-length artifact: own answers are ~2× longer (median 394 vs 185 tokens), but the instruct gain survives within logged-length tertiles (+0.04 / +0.07 / +0.07) — per-fold points and layer-14/18 + length panels: [folds](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b610115c89304bebecd7d52b718fa335bb53f9b7/figures/issue_825/onpolicy_turn_depth_folds.png), [exploratory panels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b610115c89304bebecd7d52b718fa335bb53f9b7/figures/issue_825/onpolicy_turn_depth_exploratory.png)
- Caveats: one generation draw per turn, so deep-turn cells (n ≲ 60) are exploratory; 60 of 2,572 pretrained pairs (28 at turn 1) were dropped pair-safe from both provenance arms where a tokenizer merge at the plain-text answer boundary shifts the span position; the single-turn anchor recipe differs slightly (5-fold vs 6-fold here); instruct dropped 1 pair for context-window overflow

## Next Steps

- Check if the per-turn mapping remains the SAME map across turns (cross-turn transfer + operator similarity), with generated deep conversations to reach ~5000 samples per turn depth
- Tie base model mapping -> instruct model mapping to some form of elicitation vs teaching
    - i.e. if the mapping in the instruct model can't be reconstructed from the base model's mapping then there must be some additional skills being taught to the instruct model that aren't present in the base model
    - could check if doing RLVR changes the base -> instruct mapping more (the model I'm using is too old to have done RLVR)

---

**Methodology:** [docs/methodology/issue_825.md](https://github.com/superkaiba/explore-persona-space/blob/873c6ea21e87715b1ed44fd08642379e1ef4d941/docs/methodology/issue_825.md) · [gist](https://gist.github.com/superkaiba/cd6dd04dbacb42587b5458b918710cdb)

**Context:** Thomas-authored results summary (verbatim from chat, 2026-07-15) saved as the clean result, superseding the analyzer-authored v4 body — preserved at `tasks/<status>/825/artifacts/analyzer-clean-result-v4-2026-07-15.md` and in git history. Figures SHA-pinned to `43d7a7e3ea` (main): #825 figure set, #1310 per-character stories, and the cross-issue summary set `figures/summaries/context_answer_map/`. Result 3 updated 2026-07-15 by the `naturalistic-single-turn` same-issue follow-up (verbatim originating prompt: "run it as inline followup as parallelized and vectorized as possible"): the draft template-comparison figure and its placeholder numbers are replaced by the final Track-S refit — data at `eval_results/issue_825/naturalistic-single-turn/` (issue-825 branch @ `cf2b8a8d34`, recovered from GCP attempt att-20260715-072622 after an upload-phase HF 429 crash; fits complete, upload interrupted), round figures SHA-pinned to `c3a01240f0` (issue-825 branch), figure script `scripts/issue825_naturalistic_s_figures.py`. Result 6 added 2026-07-15 by the `onpolicy-turn-depth-map` same-issue follow-up (verbatim originating prompt: "reerun with on policy answers for the multi-turn"): data at `eval_results/issue_825/onpolicy_turn_depth/results.json`, round figures SHA-pinned to `b610115c89` (issue-825 branch); generation + capture `scripts/issue825_onpolicy_turn_depth_gpu.py`, fits + figures `scripts/issue825_onpolicy_turn_depth_fit.py`; reused [#1092](https://eps.superkaiba.com/tasks/1092) banked context/answer tensors at data-repo revision `9dd650deef` (new capture: own-answer spans + prefix positions); registered decision read: provenance contributes (pooled and turn-1 own−logged deltas positive, CIs excluding zero) but turn-1 own-answer R² 0.346 sits below the 0.40 recovery threshold keyed to the 0.476 anchor — remaining gap attributed to the corpus.

