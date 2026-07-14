# Result: The context→answer mapping is inherited from pretraining and specific to the assistant turn in the chat regime

## Motivation

* An earlier experiment found a linear mapping from a single context vector to the mean answer vector in the instruct model (R² ~0.7, [#779](https://eps.superkaiba.com/tasks/779))
* We wanted to characterize where that mapping comes from and what carries it:
    * does it exist in the pretrained-only model, and what does post-training change?
    * does it require the chat-template tokens?
    * is it a general character→dialogue mechanism (fiction), or generic next-span prediction, or something chat-specific?
    * which part of the input carries it, and how far ahead does it reach?

## TLDR

- **The mapping already exists almost entirely in the pretrained base model:**
    - held-out R² 0.588 vs 0.673 at the shared best layer — **~87% of instruct prediction strength**
- **Post-training reparameterizes the existing mapping by a general linear map (NOT a rotation):**
    - a general-linear basis change on each side reconstructs both maps to **100% of ceiling** (base→instruct 1.000×, instruct→base 1.002×); the orthogonal (rotation-only) version **fails** (−0.54 / 0.59), and the non-orthogonality localizes to the **context side** (answer spaces are near-orthogonally alignable, context spaces are not)
- **The mapping holds up without chat-template tokens:**
    - the template gates *linear* decodability only: the MLP recovers template-free cells at template level (0.534/0.499 vs 0.557/0.487), and the base model reads pure `User:`/`Assistant:` transcripts at R² 0.71–0.74
- **The mapping does NOT hold up for generic stories (off-policy or on-policy):**
    - chat↔fiction transfer ≤**0.05** of ceiling vs the 0.5 same-map bar; author-blocked real-novel R² **−0.065**; on-policy model-written stories fail too (per-story R² negative in 270/270)
- **The mapping does NOT reduce to generic next-span prediction:**
    - a persona-free separator→span control transfers only **5.7% (base) / 10.9% (instruct)** of the chat map
- **The mapping is assistant-turn-specific:**
    - the user's next turn is linearly unpredictable (ridge R² negative in all 12 cells, incl. real logged human turns) and only weakly nonlinearly predictable (MLP 0.19–0.23, ~2.4× below the assistant map)
- **In richer two-turn conversations the linear read collapses while the information survives:**
    - ridge +0.08 to −0.46 vs MLP 0.49–0.56; predicting the turn-after-next fails to beat a topic-persistence baseline

## Shared methodology

Everything below uses one rig unless a result says otherwise:

- **Models:** Qwen2.5-7B (pretrained base) vs Qwen2.5-7B-Instruct, always compared on identical text (teacher-forced activation capture over fixed sequences, bf16, all 28 layers)
- **The map:** $v_C \to v_A$ — $v_C$ = a single context summary (activation at the end of the context: the assistant-header position for chat renders), $v_A$ = the mean activation over the answer span (the recipe that survived the exhaustive 46-summary sweep in [#920](https://eps.superkaiba.com/tasks/920))
- **Estimator:** per-layer GCV Gram ridge, K=5 group-level cross-validation folds (seed 0), frozen read layers {14, 18, 19, 26} with the headline at 19; 20 selection-symmetric shuffle nulls + 1,000-draw group bootstrap per cell; a 1-hidden-layer MLP probe (full-dim input, PCA-64 target head) as the nonlinearity check
- **Cross-regime transfer:** a map fitted in regime A applied to regime B, recentered and power-matched, reported as a fraction of B's within-regime ceiling; ≥0.5 of ceiling in both directions = "same map" (pre-registered bar)
- **Metric:** held-out R² (variance-weighted over the 3,584 dims). Not cosine — all $v_A$ share a large common component, so even predicting the mean answer vector scores cosine ~0.98
- **Datasets** (per-arm provenance): LMSYS-5000 single-turn (real prompts; answers sampled on-policy from instruct, T=1.0/top_p=0.95/seed 42, one answer set teacher-forced through both models); 2,000 two-turn conversations (real first turns; assistant turns instruct-generated greedy; second user turn in three provenance arms — Haiku-written / self-written / real logged); 21,193 real prefix×query rows from WildChat/LMSYS ([#1092](https://eps.superkaiba.com/tasks/1092); own-answer cells generated greedy by each model, base in naturalistic transcript format); 28 real novels (PDNC, gold speaker attribution, off-policy human text) + 1,035 model-written story pairs ([#931](https://eps.superkaiba.com/tasks/931)); 3,600 WikiText separator→span control pairs
- **Dashboard** (all experiments, links, examples): https://htmlpreview.github.io/?https://gist.githubusercontent.com/superkaiba/058d29d026937df7114d65bc3b06570e/raw/context_answer_map_dashboard.html

## Results:

### _Result 1: The mapping already exists almost entirely in the pretrained base model_

**Methodology (this result):** the #779 single-turn rig replicated exactly on both models — the same 5,000 conversations (identical instruct-written response text) teacher-forced through base and instruct, same ridge recipe, fitted independently per model ([#825](https://eps.superkaiba.com/tasks/825) Track S; supporting reads from [#810](https://eps.superkaiba.com/tasks/810) and #1092).

![single-turn layer curves, base vs instruct](https://raw.githubusercontent.com/superkaiba/explore-persona-space/745e62f4b6cbb510d529a8778cb0137b98522cc8/figures/issue_825/s_track_layer_curves.png)

**Takeaways:**

* Base **0.588** vs instruct **0.673** at the shared best layer 19 — **87.3%** of instruct strength (shuffled-pairing nulls ≈ 0, global-mean baseline −0.002)
* So post-training does not create the mapping; it adds ~13%. Caveat: pretraining corpora contain chat-formatted text, so "inherited" doesn't rule out chat-data exposure during pretraining
* The base map is genre-general (#810: skill +0.796 on UltraChat vs +0.800 on a misalignment pool) and replicates at 4× scale on real conversations (#1092: base context-arm R² 0.71–0.74)

### _Result 2: Post-training reparameterizes the mapping by a general linear map — not a rotation, and the change lives on the context side_

**Methodology (this result):** first, two swap tests on the single-turn stores (both models, identical text): *representation-swap* (fit a fresh map from base contexts to instruct answers — does the information transfer?) and *map-swap* (apply model A's fitted weights to model B's activations — do the coordinates transfer?). Then the **direct composition test**: fit explicit alignment maps between the models' context spaces (A_ctx) and answer spaces (A_ans), and check whether one basis change reconstructs the other model's map held-out — `M_inst ≈ A_ans ∘ M_base ∘ A_ctx⁻¹` — under a **general-linear** A vs an **orthogonal (Procrustes)** A. Random-rotation null for the weight-space Procrustes cosine (n=100); residual-rank read on the aligned difference.

![composition R²: general-linear vs orthogonal basis change](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ccee9ad8d6d83b73c86c93f3dbdc81dabc403a4a/figures/issue_825/map_alignment_composition_r2.png)

**Takeaways:**

* The information transfers, the coordinates don't: representation-swap preserves prediction (**0.587 / 0.662** at layer 19, vs 0.673/0.588 within-model); map-swap fails (**−0.066 / −0.170**)
* **A general-linear basis change is the whole story:** `A_ans ∘ M_base ∘ A_ctx⁻¹` reconstructs both maps to **100% of ceiling** (base→instruct 1.000×, instruct→base 1.002×) — the two maps are the same function in different coordinates
* **It is NOT a rotation:** the orthogonal-constrained version fails (composition fraction **−0.54 / 0.59**), and the non-orthogonality **localizes to the context side** — answer spaces are near-orthogonally alignable (A_ans held-out R² 0.902 linear vs 0.893 orthogonal), context spaces are not (A_ctx 0.619 linear vs 0.308 orthogonal). So post-training re-bases how the *context* is read, not just how it is rotated
* The 0.686 weight-space Procrustes cosine is real (z≈2412 above the random-rotation null, mean ~0) but **overstates functional agreement** — it reflects partial orthogonal alignability while the held-out orthogonal composition is negative; the residual difference is diffuse and high-rank (participation ratio 1013/3584, top-50 SV share 0.45), not low-rank
* The change is specific to the chat map: the persona-free separator-control map (Result 5) IS shared across models (weight cosine 0.86–0.90)
* An apparent base-vs-instruct gap in predicting own continuations was a decoding artifact: greedy D 0.590/0.428 collapses to 0.031/0.086 under sampled decoding — the substrates agree at matched decoding

### _Result 3: The mapping holds up without chat-template tokens_

**Methodology (this result):** the 2,000 two-turn conversations rendered TWO ways — chat template vs a naturalistic `User:`/`Assistant:` plain-text transcript (identical words, no special tokens) — with ridge + MLP fit per (model × format) cell; plus #1092, where the base model reads pure naturalistic transcripts at scale. (Running: the single-turn cells re-rendered naturalistically, both models — closes the strong-regime instruct cell.)

![two-turn cells by model and format](https://raw.githubusercontent.com/superkaiba/explore-persona-space/745e62f4b6cbb510d529a8778cb0137b98522cc8/figures/issue_825/within_cell_best_frozen_r2.png)

**Takeaways:**

* Stripping the template hurts the *linear* map (instruct ridge +0.076 chat → −0.078 naturalistic) but the **MLP recovers the naturalistic cells at chat level** (0.534/0.499 vs chat siblings 0.557/0.487) — the template tokens make the map linearly decodable; they don't carry it
* At scale on real conversations the base model reads template-free transcripts at R² **0.714–0.742** (#1092) — no template anywhere in that rig
* The remaining untested cell (instruct on template-free text in a strong regime) is the refit currently running on #825

### _Result 4: The mapping does NOT hold up for generic stories_

**Methodology (this result):** the identical estimator refit on fiction with no chat template — X = a character's introduction span, Y = the mean activation over that character's attributed dialogue. Two arms: 28 real novels (PDNC corpus, gold attribution — off-policy human text) and 743 model-written multi-character stories (on-policy, T=1.0). Cross-regime transfer against the chat map; author-blocked folds (19 authors) as the honest generalization read; a character-swap contrast (predict A's dialogue from B's intro, shared paired bootstrap) for character identity ([#931](https://eps.superkaiba.com/tasks/931)).

![cross-regime transfer vs the 0.5 same-map bar](https://raw.githubusercontent.com/superkaiba/explore-persona-space/024e2a22c7a23c172b792c43ff00678ed4cefcea/figures/summaries/context_answer_map/transfer_specificity.png)

**Takeaways:**

* No transfer in either direction: chat↔fiction peaks at ~**0.05** of the within-regime ceiling — an order of magnitude below the 0.5 same-map bar. The on-policy stories arm fails too, so this is not an off-policy artifact
* The within-fiction signal is mostly **author style**: author-blocked folds send the real-novel R² from +0.17 to **−0.065** (an author-level component ≈0.23 of the pooled R²); per-story R² is negative in 270/270 stories
* Character identity is a small real residue: swapping the predicted character costs R² 0.076 (novels) / 0.046 (stories), 91–93% surviving distance partialling, correct-beats-swap 28/28 novels

### _Result 5: The mapping does NOT reduce to generic next-span prediction_

**Methodology (this result):** a persona-free control with the same machinery — predict the span after a sentence-final separator token from that token's activation, on WikiText articles (3,600 pairs), fitted within each model and transferred to the chat map with the same recentered power-matched protocol (#825 rounds 6–8).

![separator control, base vs instruct](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b5959c212812e358c667a8e955575822ccfc7075/figures/issue_825/base_sep_control_hero.png)

**Takeaways:**

* The separator→span control transfers only **5.7% (base) / 10.9% (instruct)** of the chat map's information — generic "predict what comes after this boundary" is not what the chat map is
* Unlike the chat map, the control map is **model-shared** (base↔instruct weight cosine 0.86–0.90): post-training left generic next-span structure alone and changed the chat map specifically (Result 2)
* Together with Result 4 this pins the mapping to the assistant-turn conversational structure, not to raw-text statistics

### _Result 6: The mapping is assistant-turn-specific_

**Methodology (this result):** the two-turn conversations with the *user's next turn* as the prediction target, under three user-text provenances (Haiku-written, self-written by the measured model, real logged human turns — 2,000 fully-logged conversations); round 5 refit both roles of every conversation on shared rows and folds so the role contrast is formally paired (1,000 shared bootstrap resamples).

![paired role contrast, all 12 pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5f4334711c39eedea65b580ed51ec09c15887764/figures/issue_825/role_contrast_paired_delta.png)

**Takeaways:**

* The user's next turn is **linearly unpredictable**: ridge R² negative in all 12 (provenance × model × format) cells (−0.6 to −1.8 at layer 19) — including on real logged human turns, so it's not a synthetic-text artifact
* And only **weakly nonlinearly predictable**: MLP 0.19–0.23 (~50 null-band-widths above chance, but ~2.4× below the assistant map)
* The assistant map beats the user map in **all 24** paired reads (ridge Δ +0.36 to +1.53, every 95% CI clear of zero)

### _Result 7: The two-turn regime breaks the linear read, not the map — and the map doesn't reach past the next turn_

**Methodology (this result):** the two-turn within-cells (predict the upcoming assistant turn inside a 2-turn conversation) — ridge vs MLP per cell; the cross-role cells (predict the turn-after-next: v(u2) from before a1, v(a2) from before u2) against a topic-persistence baseline (predict v(u2) from v(u1)), paired per conversation; [#922](https://eps.superkaiba.com/tasks/922) for the within-answer horizon.

![ridge vs MLP, all 10 probed cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a302438d6890ec953c39be72a5f0e9865ad695fe/figures/issue_825/ridge_vs_mlp_all_cells.png)

**Takeaways:**

* In two-turn conversations the linear map collapses (ridge +0.076 to −0.461 across the four assistant cells) while the MLP holds at **0.49–0.56** — regime fragility of the *linear* read, not loss of information. Not just sample size: single-turn at matched n=2,000 still reads 0.321
* Predicting the **turn after next** fails linearly: the cross-role map loses to a topic-persistence baseline in 13 of 16 frozen-layer reads (paired Δ −0.2 to −0.7; one exception: pretrained assistant→user at layers 19/26, +0.12/+0.23). No MLP was run on these cells
* *Within* the answer, the horizon is real: a rolled per-layer affine map forecasts answer-time activations from the prompt alone through **32 tokens** (#922, HIGH confidence)
* Realism costs a little more: real logged conversations depress the two-turn assistant map (MLP 0.32–0.38 vs 0.49–0.56 on synthetic twins)

## Next steps:

- (running) single-turn naturalistic refit on #825 — the last format×model cell (instruct, template-free, strong regime), with a paired chat-vs-naturalistic contrast on shared folds
- (done, Result 2) the direct basis-change test settled it: general-linear reparameterization, not a rotation, with the change on the context side
- MLP probe on the cross-role (turn-after-next) cells — the linear-only verdict there mirrors the user-turn case, where the MLP found a weak map the ridge missed
- Use the rotation read quantitatively: how big is the basis change, is it low-rank, does it concentrate at the map's peak layers
