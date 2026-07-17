# Result: The mapping already exists in the pretrained model — post-training rotates its coordinates rather than creating it, and it is answer-specific + chat-specific, not a general character→behavior map

## Motivation
* We showed in a previous experiment that [[There is a linear mapping between single context vector and answer summaries]]
* We wanted to see
    * if this mapping holds in the pretrained-only model vs the instruct model
    * how this mapping changes from the pretrained to the instruct model
    * if it requires the chat template or holds for more general character + dialogue pairs

## TLDR

- The mapping already exists in the pretrained base model, at 87% of instruct strength:
    - held-out R² 0.588 (pretrained) vs 0.673 (instruct) at layer 19; shuffle nulls ≈ −0.02
    - so post-training did not create the map
- Post-training **rotates the map's coordinates without changing its information**:
    - applying one model's fitted map to the other model's activations fails outright (R² −0.066 / −0.170)
    - but re-FITTING a map from one model's context activations to the *other* model's answer activations works as well as within-model (0.587 / 0.662)
    - i.e. the information the map reads survives post-training; only the coordinate system it is written in moves
- The mapping is **answer-specific in both models**, not generic "predict the following span" structure:
    - a control map (sentence-final punctuation token → mean activation until the next punctuation, on frozen WikiText) transfers only **5.7%** (pretrained) / **10.9%** (instruct) of the chat map's information
- The punctuation control map is the one thing that does NOT rotate: it is essentially the **same map in both models** (weight cosine 0.86–0.90; random reference ≈ 0)
    - so post-training rotated the answer map specifically, while leaving the generic span-prediction machinery untouched
- Replacing the chat template with a plain transcript (`User: ... \n\n Assistant: ...`, colon anchors) removes the *linear* readability but not the map:
    - instruct two-turn ridge drops +0.076 (chat) → −0.078 (naturalistic), but an MLP reads the naturalistic cells at 0.534 / 0.499 ≈ the chat cells (0.557 / 0.487)
    - so the template gates WHERE the map is linearly written, not whether it exists — same flavor as the rotation result
- The mapping does **not generalize to fiction** character→dialogue pairs:
    - chat→fiction transfer peaks at **+0.05 of ceiling** against a pre-registered 0.5 "same map" bar
    - what fiction hosts instead is mostly an **author-level** map (author-blocked folds send the novels R² from +0.17 to −0.065), plus a small but real **character-identity** component (swapping which character you predict costs R² 0.076 in novels / 0.046 in stories; correct beats swap in 28 of 28 novels)
- Caveat under test right now: the chat answers are the model's OWN generations while the punctuation-control spans are exogenous text it merely read — a model's own upcoming text is much easier to predict from its state, so part of "answer-specific" could really be "self-generated-text-specific". An on-policy punctuation control (each model continues WikiText itself, then we anchor on punctuation inside its own text) is running in both models now.

## Methodology

- Models: **Qwen-2.5-7B** (pretrained base) and **Qwen-2.5-7B-Instruct** — the previous experiment was instruct-only
- Datasets:
    - **LMSYS-5000 × 2 models**: the same 5,000 real user prompts as the previous experiment, chat template, answers generated on-policy separately by EACH model (each model's map predicts its own answers)
    - **Punctuation-span control (frozen text)**: 3,600 pairs from 600 WikiText-103 articles — anchor = a sentence-final punctuation token {`.`, `!`, `?`}, target = mean activation over the following span until the next such token (spans 8–256 tokens). Byte-identical pairs fed to both models; no generation anywhere (pairs pinned at HF rev `9534b998`, `issue931_story_map/raw_completions/pairs_meta`)
    - **Fiction, real novels**: 28 dialogue-rich novels with gold speaker attribution, fed as raw text with NO chat template — 1,982 (character-introduction span → that character's attributed dialogue) pairs
    - **Fiction, model-written stories**: 1,035 pairs across 743 multi-character stories the instruct model wrote itself under the chat template
    - **Chat-template vs naturalistic format (two-turn cells)**: the same 2,000 conversations rendered two ways — chat template, and a plain transcript `User: <text>\n\nAssistant: <text>\n\n` with the role-header colon as the anchor slot; content tokens gated token-identical across formats (interior-core BPE identity ≤10% mismatch)
- Computed quantities (same as the previous experiment):
    - $v_C$: activation at the last prompt token (fiction: mean over the character's introduction span; control: the punctuation-token activation)
    - $v_A$: mean activation over the answer span (fiction: over the character's attributed dialogue; control: over the span until the next punctuation)
- Predictors: GCV ridge per layer (5 group folds, seed 0), a rotated random-projection variant, MLP secondary at frozen layers {14, 18, 19, 26}; group-blocked shuffle nulls (20 draws) + 1,000-draw group bootstrap
- Cross-model reads (base ↔ instruct):
    - **map-swap**: take the map FITTED on model A and apply it to model B's activations — tests whether the same coefficients work
    - **representation-swap**: re-fit a fresh map from model A's context activations to model B's answer activations — tests whether the *information* is present regardless of coordinates
    - weight-space similarity of the two fitted maps (per-output-dimension coefficient cosine, principal angles, Procrustes), 100 permutation nulls
- Everything gated before decision reads: instruct anchors re-fit within ±0.01 of committed values, the base chat cell reproduced its committed 0.588 exactly, tokenizer-identity checks
- Task pages with full per-round provenance: https://eps.superkaiba.com/tasks/825 (base/instruct + control) and https://eps.superkaiba.com/tasks/931 (fiction)

## Results:

### _Result 1: The mapping already exists in the pretrained base model (R² 0.588 vs 0.673 instruct, 87% of instruct strength)_

The first question was whether the context→answer map is something post-training installs, or something already present in the pretrained model. I ran the identical single-turn rig on both models — same 5,000 LMSYS prompts, each model predicting its own on-policy answers.

**Plot: held-out R² by layer, pretrained vs instruct, single-turn**

![base vs instruct layer curves](https://raw.githubusercontent.com/superkaiba/explore-persona-space/745e62f4b6cbb510d529a8778cb0137b98522cc8/figures/issue_825/s_track_layer_curves.png)

**Takeaways:**

* The map exists in the pretrained model: R² 0.588 [0.579, 0.598] vs instruct 0.673 [0.666, 0.681] at the frozen layer 19; shuffle nulls ≈ −0.02 at every layer
* So the map is not created by an explicit post-training stage — post-training adds ~13% relative strength
* One honest caveat: pretraining corpora contain chat-formatted text, so "no chat-data exposure" is not ruled out — only "no dedicated post-training stage needed"
* This is the single-turn picture. In the harder two-turn cells the *linear* map is much weaker in the base model (chat-format ridge −0.46 base vs +0.08 instruct) but an MLP recovers both (0.49 base / 0.56 instruct) — so the deeper-context version of the map is nonlinearly present in both models too

### _Result 2: Post-training rotates the map's coordinates without changing its information_

Given the map exists in both models, the natural question is whether it's the *same* map. Two swap tests answer different halves of this: map-swap (apply model A's fitted coefficients to model B) tests the coordinates; representation-swap (re-fit across models) tests the information.

**Plot: within-model vs map-swap vs representation-swap at layer 19**

![cross-model transfer](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4d03165dd8a1773790a1b501b3699c1a6e08e584/figures/issue_825/crossmodel_transfer_r2.png)

**Plot: weight-space similarity of the two fitted maps**

![weight-space similarity](https://raw.githubusercontent.com/superkaiba/explore-persona-space/37a5ba8a6a0293e90d400d9c4a069f3a6fe57d3d/figures/issue_825/crossmodel_map_similarity.png)

**Takeaways:**

* Map-swap fails completely (R² −0.066 / −0.170) while representation-swap recovers within-model performance (0.587 / 0.662) — the information is preserved across post-training, the coordinates are not
* The intuitive reading: post-training moved WHERE the answer-relevant information sits in the residual stream (a big rotation of the read-out basis), not WHETHER it's there
* Weight space agrees the maps are related but far from identical: per-output-dimension coefficient cosine has median 0.207 (random reference 0.000), the top-50 principal-angle read barely clears its random-map reference (0.856 vs 0.835), and the Procrustes-aligned cosine (0.686) is exploratory — descriptive only, single seed, no mechanism claim
* Follow-up worth doing: explicitly align the two answer spaces (Procrustes on paired activations) and test whether the aligned map transfers — that would upgrade "rotation" from a description to a tested mechanism

### _Result 3: The mapping is answer-specific in both models — and the punctuation control map is the one map that does NOT rotate_

A worry about Results 1–2: maybe "context state predicts following-span mean" is just generic structure any span boundary has, and the answer map isn't special. The control: fit the identical machinery anchored on sentence-final punctuation tokens in frozen WikiText (punctuation activation → mean activation until the next punctuation), identical pinned pairs in both models, then measure how much of the chat map's information the control map carries (transfer, recentered, as a fraction of the chat ceiling).

**Plot: chat map vs punctuation-control map, pretrained vs instruct, layer 19**

![separator control hero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b5959c212812e358c667a8e955575822ccfc7075/figures/issue_825/base_sep_control_hero.png)

**Plot (raw companion): per-article-group control R², instruct vs pretrained**

![per-group scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b5959c212812e358c667a8e955575822ccfc7075/figures/issue_825/base_sep_pergroup_scatter.png)

**Plot: weight cosine between the two models' punctuation-control maps**

![separator weight cosine](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b5959c212812e358c667a8e955575822ccfc7075/figures/issue_825/base_sep_weightcos.png)

**Takeaways:**

* The punctuation→span control transfers only 5.7% of the base chat map's information (instruct: 10.9%) — far below the pre-registered 0.5 "same map" line, so the chat map is not generic span-prediction structure, on either model
* The control behaves the same way in both models (within-strength ratio 0.62 base vs 0.52 instruct, inside the pre-registered ±0.10 mirror margin) — the specificity conclusion is substrate-stable
* The striking contrast: the control map is essentially the SAME map in both models (weight cosine 0.86–0.90 across the frozen layers; random ≈ 0) while the chat answer map rotates (Result 2). Post-training moved the answer read-out specifically and left the generic span machinery alone
* Two caveats to keep honest:
    * plain ridge fails pathologically on the control cells in both models (R² −2.9 to −3.5; positive only at layers 0–3) — the control's within-strength numbers come from the rotated variant (0.363 base / 0.349 instruct) and the MLP (0.311 / 0.299), and the failure itself is documented, not hidden
    * the chat answers are self-generated, the control spans are not — the on-policy punctuation control now running is the arm that isolates this (see Next steps)

### _Result 4: The mapping does not generalize to fiction — fiction hosts a mostly author-level map with a small character-identity component_

The last motivation question: is this a general character→behavior map, or does it need the chat setup? We re-fit the identical estimator on raw-text fiction — real novels with gold speaker attribution fed with no chat template (character introduction → that character's dialogue), and multi-character stories the instruct model wrote itself — then measured transfer against the chat map.

**Plot: cross-regime transfer fractions of ceiling (chat ↔ novels ↔ stories)**

![fiction transfer matrix](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc1c447b6ced971dd9321a764476b1c272598372/figures/issue_931/hero2_transfer_matrix.png)

**Plot: what survives honest folds — novel-blocked vs author-blocked**

![author-blocked folds](https://raw.githubusercontent.com/superkaiba/explore-persona-space/174e51d6a74d8c5b2144e708bd2b8aec3b5b16e3/figures/issue_931/author_blocked_folds.png)

**Plot: the character-identity component (correct vs swapped character, per novel)**

![character swap per novel](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc1c447b6ced971dd9321a764476b1c272598372/figures/issue_931/delta_char_per_novel.png)

**Takeaways:**

* The chat map does not carry over: chat→fiction transfer peaks at +0.05 of ceiling against the 0.5 bar — an order of magnitude short. The map we found is a chat-regime object, not a universal character→behavior mechanism
* The weak fiction map that does exist is mostly not about characters: blocking folds by AUTHOR (so the fit never sees the test author) sends the novels R² from +0.17 to −0.065, and a decomposition puts the author-level component at ≈0.23 — the map was largely recognizing authors/styles, not predicting characters
* Character identity is a small real component on top: swapping which character's dialogue you predict costs R² 0.076 (novels) / 0.046 (stories), 91–93% of which survives partialling out how far apart the characters' context vectors are; correct beats swap in 28 of 28 novels
* No chat↔fiction similarity read exceeds its persona-free control (prediction CKA 0.227 vs 0.624 for a preceding-sentence control) — even the above-null similarity between regimes looks generic
* Same estimator-fragility caveat as Result 3: ridge fails on the WikiText control cells here too (−3.2 / −3.5 vs +0.35 rotated), so the control comparisons ride the rotated/MLP reads

### _Result 5: Replacing the chat template with a plain transcript removes the linear readability, not the map_

The original ask for this line included "try one with the chat template and one replacing it with a more naturalistic format (e.g. semicolon)". The implemented version strips the chat template entirely and renders the same conversations as a plain transcript — `User: <text>\n\nAssistant: <text>\n\n` — with the colon at the end of the role header as the anchor slot (the analogue of the assistant-header position in the chat render). Content tokens are gated token-identical across the two renders, so the format is the only thing that changes. This axis ran factorially over both models and both roles in the two-turn cells.

**Plot: best frozen-layer R² per two-turn cell, by model and format (ridge)**

![two-turn cells by model and format](https://raw.githubusercontent.com/superkaiba/explore-persona-space/745e62f4b6cbb510d529a8778cb0137b98522cc8/figures/issue_825/within_cell_best_frozen_r2.png)

**Plot: ridge vs MLP for all ten probed cells — the naturalistic cells come back under the MLP**

![ridge vs MLP all cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a302438d6890ec953c39be72a5f0e9865ad695fe/figures/issue_825/ridge_vs_mlp_all_cells.png)

**Takeaways:**

* Stripping the template costs the instruct model its *linear* map in the two-turn cells: ridge +0.076 [0.052, 0.098] (chat) → −0.078 [−0.106, −0.050] (naturalistic) at layer 19
* On the pretrained model both formats are linearly floored (−0.461 chat / −0.390 naturalistic), with the naturalistic side slightly less bad — directionally what you'd expect if raw text is base's native regime, though both are below zero
* The map itself survives the template removal: the MLP reads the naturalistic assistant cells at 0.534 / 0.499, essentially matching the chat cells (0.557 / 0.487) — **the format gates where the map is linearly written, not whether it exists**
* This rhymes with Result 2: the chat template, like post-training, changes the coordinates the map is written in rather than the information it carries
* Provenance caveat: the assistant text is the model's own (each model generated its own first assistant turn, greedy), but it was generated once under the CHAT render and then re-rendered into the transcript — so the naturalistic cells are teacher-forced on the model's own chat-generated text, not text produced while actually in a transcript context. A fully on-policy naturalistic arm (generate the answer *in* the transcript regime) hasn't run — the same family of confound as the punctuation control's frozen text, milder because the text is at least self-generated; the on-policy punctuation round now running probes the sharpest version of it
* Note the exact manipulation: role headers with colons, not literal semicolons — a fully role-free delimiter render (e.g. turns separated by bare semicolons, no `User:`/`Assistant:` words) has not been run, and would tell us whether the role words or just *some* consistent delimiter structure is what the linear read-out needs

## Next steps:
- (running) **on-policy punctuation control in both models** — each model continues WikiText prefixes itself, we anchor on punctuation inside its own generated text. If the on-policy control map jumps toward the chat map's strength, Result 3 weakens from "answer-specific" to "self-generated-text-specific"; if it stays near the frozen-text control, answer-specificity survives the strongest version of the objection
- Test the rotation mechanically: Procrustes-align the two models' answer spaces on paired activations and check whether the aligned map transfers (turns Result 2's "rotation" description into a testable claim)
- The isolating fiction control: regenerate the first answers with the measured model on the same real conversations (named in the fiction experiment as the follow-up that would attribute its real-conversation break)
- Extend the base arm beyond single-turn: the two-turn / user-turn cells on base currently exist only in the parent rounds; a base-side nonlinear user-map read would complete the substrate comparison
- Both tasks are parked for promotion: https://eps.superkaiba.com/tasks/825 and https://eps.superkaiba.com/tasks/931

---
*Sources: task #825 (rounds 1–6, incl. the cross-model swap analysis and the base punctuation control) and task #931 (fiction generalization). All numbers verbatim from the committed clean-result bodies; figures pinned to their committed SHAs. Confidence: MODERATE on both tasks (single seed throughout; the base transfer read uses the full-n convention because the matched-n denominator was unevaluable — base chat ceiling collapses negative at n=1,982 in all 5 draws).*
