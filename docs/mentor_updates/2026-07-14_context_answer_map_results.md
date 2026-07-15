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
    - held-out R² 0.588 for base vs 0.673 for instruct at the shared best layer — **~87% of instruct prediction strength**
- **Post-training reparameterizes the existing mapping by a general linear map (not a rotation):**
    - base's map, run through a fitted general-linear change of coordinates, reconstructs the instruct map to **100% of the instruct held-out ceiling** (composition R² 0.673); a rotation (orthogonal change of coordinates) fails (composition −0.36). The change lives on the **context side** (context vectors ~62% linearly relatable base↔instruct vs ~90% for answers).
- **The mapping holds up without chat-template tokens:**
    - $R^2 \approx 0.71$–$0.74$ for plain "User:/Assistant:" transcripts in the base model
    - $R^2 \approx X_1$ (instruct, single-turn no-template) — **pending** (the direct Track-S re-run is extracting now; the #1092 instruct cells are chat-templated so can't fill this). On the two-turn task the instruct *linear* map is template-gated (ridge +0.076 → −0.078) while the nonlinear MLP is format-invariant (0.56 → 0.53).
- **The mapping does not hold up for the user turn:**
    - the user's next turn is linearly unpredictable (ridge R² negative for both real human turns and model-generated user turns) and only weakly nonlinearly predictable (MLP 0.19–0.23)
- **The mapping does NOT hold up for generic stories (off-policy or on-policy):**
    - re-training the mapping on generic stories gets $R^2 \approx$ **0.16** for on-policy model-generated stories and **−0.07** for off-policy stories from the internet — both at/near the shuffled floor (0.06–0.08), vs 0.67 for chat
- **The mapping is not just generic next-span prediction:**
    - a mapping from a separator token to the span before the next separator (generic text) transfers to the chat map at only $R^2 \approx$ **5.7% (base) / 10.9% (instruct)** of the within-regime ceiling (vs a 0.5 same-map bar); its own within-model R² is near-floor at layer 19
- **When predicting the assistant's response in two turns, the linear mapping fails, but a nonlinear mapping survives:**
    - ridge regression $R^2 \approx −0.46$ to $+0.08$ vs MLP $R^2 \approx 0.49$–$0.56$

## Methodology

- **Models:** Qwen2.5-7B (pretrained base) vs Qwen2.5-7B-Instruct
- $v_C$: activation at the end of the context (taken at best layer from prior experiments)
- $v_A$: the mean activation over the answer span (taken at best layer from prior experiments)
- **Estimator:** ridge regression for linear, MLP for nonlinear
- **Metric:** held-out R² (variance-weighted over the 3,584 dims), 5-fold cross-validation, shared best layer 19

## Results:

### _Result 1: The mapping already exists almost entirely in the pretrained base model_

I first wanted to see if the mapping already exists in the base model.

**Methodology (this result):**
- 5000 LMSYS user turns
- answers generated either by base model or instruct model
- 5-fold cross validation across 4000 samples (chat templated data in both base **and** instruct model)
- compute mapping in base model
- compute mapping in instruct model
- Compare $R^2$ for both at all layers
- Baseline = shuffled context/answer pairings

![single-turn layer curves, base vs instruct](https://raw.githubusercontent.com/superkaiba/explore-persona-space/745e62f4b6cbb510d529a8778cb0137b98522cc8/figures/issue_825/s_track_layer_curves.png)

**Takeaways:**

- The mapping already mostly exists in the base model:
    * Base **0.588** vs instruct **0.673** at the shared best layer 19 — **87.3%** of instruct strength (both far above the shuffled floor ~0.06)

### _Result 2: Post-training reparameterizes the existing map by a general linear map_

I then wanted to see whether post-training builds a *new* map or just re-expresses the existing one in different coordinates.

**Methodology (this result):**
- Fit the ridge operator $M$ (context→answer) separately in base and instruct
- Fit a change-of-coordinates on the context side ($A_{ctx}$) and answer side ($A_{ans}$) between the two models on identical text
- Test whether base's operator, reparameterized, reproduces instruct's ($A_{ans} \circ M_{base} \circ A_{ctx}^{-1}$), under a **general-linear** vs an **orthogonal (rotation)** change of coordinates
- Decompose the base→instruct context shift into the directions the map reads vs the directions it ignores

![instruct own map vs reparameterized base map](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6ea63f4d33a981f859b1e93da42fba37a3752246/figures/issue_825/reparam_vs_instruct.png)

**Takeaways:**

- The two maps are **the same function in different coordinates, via a general linear map** — not a rotation:
    * general-linear composition reconstructs the instruct map to **100% of the instruct ceiling** (0.673); orthogonal (rotation) composition **fails (−0.36)**; scaled-rotation only reaches 83%
    * the reparameterization is **context-side**: context vectors are ~62% linearly relatable base↔instruct ($A_{ctx}$ R² 0.62) vs ~90% for answers ($A_{ans}$ R² 0.90); raw operator cosine 0.25 → 0.69 after alignment
- The context representation shifts **more than the map requires, but map-indifferently**: ~38% of the instruct context variance isn't linearly recoverable from base (beyond any linear reparameterization), and that excess splits between the map's answer-relevant and answer-ignored directions within ±2.5% of a random-subspace null (it neither targets nor avoids the map)

### _Result 3: The mapping holds up without chat-template tokens (in both base and instruct model)_

I then wanted to see if this mapping was only there because of the chat-template tokens.

**Methodology:**
- Fit the same mapping to the same conversations, replacing the chat template with just "User:" + "Assistant:"
- Plot $R^2$ with chat template vs without chat template for both instruct and pretrained model

![two-turn cells by model and format](https://raw.githubusercontent.com/superkaiba/explore-persona-space/745e62f4b6cbb510d529a8778cb0137b98522cc8/figures/issue_825/within_cell_best_frozen_r2.png)

**Takeaways:**

- The mapping **holds up** without chat-template tokens:
    * base model reads plain "User:/Assistant:" transcripts at R² **0.71–0.74** (#1092)
    * instruct single-turn no-template (X₁): **pending** — the Track-S re-run is extracting now; I'll fill it when it lands (the #1092 instruct cells are chat-templated, so they can't stand in for it)
    * on the two-turn task, format gates **only the linear** map: instruct ridge +0.076 → −0.078 between chat and naturalistic, while the MLP is format-invariant (0.56 → 0.53)

### _Result 4: The mapping does NOT hold up for generic stories, or for the user turn, or for generic next-span prediction_

I then wanted to see:
- is this just a generic "character → character output" mapping in stories (as might be predicted by PSM)?
- does this mapping hold for the user turn?
- does this mapping hold for generic next-span prediction (punctuation → span of text before next punctuation)?

**Methodology:**
- Fit mapping to:
    - character → answer in generic real stories (off-policy)
    - character → on-policy generated answer in stories generated by the model
    - user context → user answer (real data)
    - user context → user answer (model-generated)
    - separator token → span before next separator (real generic text + model-generated)
- Plot $R^2$ (and cross-regime transfer as a fraction of the within-regime ceiling) for each

![cross-regime transfer vs the 0.5 same-map bar](https://raw.githubusercontent.com/superkaiba/explore-persona-space/024e2a22c7a23c172b792c43ff00678ed4cefcea/figures/summaries/context_answer_map/transfer_specificity.png)

**Takeaways:**

* The mapping does not hold up for generic stories:
    * off-policy author-blocked novels R² = **−0.07** (n=1982); on-policy model-generated stories R² = **0.16** (n=1035) — both at/near the shuffled floor (0.06–0.08) vs 0.67 for chat, and cross-transfer to the chat map is only ~5% of ceiling
* The mapping does not hold up for the user turn:
    * ridge R² is **negative across all 12 cells** (−0.6 to −1.8 at layer 19; real human, model-generated, and Haiku-written next turns); MLP recovers only **0.19–0.23**
* The mapping does not hold up for generic next-span prediction:
    * a separator→next-span map transfers to the chat map at only **5.7% (base) / 10.9% (instruct)** of the within-regime ceiling (vs the 0.5 same-map bar); its own within-model R² is near-floor at layer 19
- I suspect this is a "per-character" mapping (though the user turn not holding up somewhat argues against that). Re-running now with a mapping trained on generations from a single fixed character in stories.

### _Result 5: The linear mapping breaks across two turns, but the nonlinear mapping holds_

I then wanted to see if you can predict the assistant answer two turns ahead from the current turn's context (both linearly and nonlinearly).

**Methodology:**
- Train linear and nonlinear mappings from the context vector to the assistant answer vector **two turns ahead**, base and instruct, chat and naturalistic formats

![ridge vs MLP, all 10 probed cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a302438d6890ec953c39be72a5f0e9865ad695fe/figures/issue_825/ridge_vs_mlp_all_cells.png)

**Takeaways:**

- The **linear** two-turn map fails but the **nonlinear** one survives:
    * two-turn assistant cells: ridge R² **+0.08 → −0.46**, MLP R² **0.49–0.56** (across all 10 probed cells the gap widens for the user-turn cells: ridge down to −1.3, MLP 0.19–0.56)
    * so the information to predict the assistant two turns out is present but **not linearly readable** — the single-turn linearity is specific to the immediate answer

## Next Steps

- **Finish Result 3:** land the Track-S single-turn no-template instruct number (X₁) — extracting now.
- **Resolve the "per-character" hypothesis:** #1310 v3 is running (prefill datagen so the base arm actually populates, + vectorized fit) to test a focused single-character context→dialogue map in base and instruct; current signal is a null (−0.12 to −0.29 instruct; base arm was empty in v2).
- **Localize what carries the map** (Motivation Q4): the reparameterization is context-side, but we haven't localized *which* part of the context — a token-position / component ablation of $v_C$.
- **Characterize the general-linear change functionally:** what does the context-side $A_{ctx}$ correspond to (scaling of which directions), and does it track a known post-training axis?
