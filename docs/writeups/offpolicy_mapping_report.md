# Result: The context→answer mapping is not policy-specific — it predicts another model's answers almost as well as its own, even where the two models genuinely diverge
<!-- report-v1 -->

## Motivation

- We found in a previous experiment ([#779](https://eps.superkaiba.com/tasks/779)) that there is a linear mapping from context to on-policy answers (test R² 0.705 [0.691, 0.719] at the validation-selected layer 19).
- This could be because:
    - the mapping is actually predicting the model's behavior
    - the mapping is just predicting "some consistent behavior"
    - the mapping is just predicting some kind of topic similarity between questions and answers
- We wanted to test this

## TLDR

- You can train an almost-as-good mapping from context to answer for off-policy text:
    - R² ≈ 0.56–0.59 for Claude-generated answers vs R² ≈ 0.60–0.63 for on-policy answers (91–98% retention across the three read-out layers)
    - not a trivial-baseline effect: swapped answers (real answers, wrong context) collapse to R² ≈ 0 (±0.01)
    - replicates on real conversations ([#1092](https://eps.superkaiba.com/tasks/1092)): 0.776 (Claude) vs 0.804 (own) — and the ordering *flips* on the pretrained reader (Claude 0.742 vs own 0.714)
- This mapping is **not** the same as the on-policy mapping — but it's mostly the same:
    - the on-policy-trained map scores Claude targets at R² 0.45–0.46 against the Claude-refit ceiling of 0.556–0.591 (a ~0.10–0.14 transfer cost); symmetric in reverse (Claude-trained map on own targets: 0.458–0.470)
    - the sharp separation is **style**: eccentric-style Claude answers refit at 0.47–0.51 but get ≈ 0 transfer from the on-policy map (−0.07..+0.05), while Claude-plain→Claude-style transfers at 0.08–0.19 — each fitted map is style-specific, and the two Claude arms share register that Qwen's map doesn't carry
    - (weight-space cosine between the fitted maps is uninformative at this n — two fits of the *same* arm on disjoint halves read cosine ≈ 0.04 — so transfer R² is the valid "same map" evidence)
- The mapping doesn't get worse where the 2 models genuinely diverge:
    - on 41 generation-verified Qwen-vs-Claude divergence pairs with entity-swapped controls, the divergence-specific penalty for the off-policy map is −0.005 (sign-flip p = 0.64; registered margin 0.05; detection ceiling 0.887, so a real effect was detectable)
    - holds in the sharpest cell: the on-policy-trained map scored on *Claude's* divergent answers shows no penalty either (−0.016 [−0.039, +0.007], p = 0.90); on model-identity pairs Claude's divergent answers are actually *more* predictable than their controls (−0.063)
    - and — after a top-up round (2026-07-15) lifted the category past its floor — on **china-politics** (31 verified pairs; Qwen refuses where Claude answers on 10 of the 18 parent pairs): arm-matched d +0.014 (Holm p = 0.38, margin not cleared), cross cell −0.001 (p = 0.54); pooled over all 72 pairs +0.003 (p = 0.38)
    - caveat: the null is well-grounded on model-identity (judged divergence 82 vs control 15; 20/20 pairs clear the original margin) and china, diluted on style-format (71 vs 49; 6/21 pairs with margin ≤ 0); refusal-boundary remains unread (2 pairs)
- Overall indicates that the mapping is something like predicting the "consistent character" of a model, and one model is able to predict the character of another model's outputs
    - with one refinement the controls force: the retention itself is fully accounted for by answer-content overlap (the on-policy answer *profile* predicts the Claude profile at R² 0.671–0.688 — better than the context does at 0.556–0.591), and the swapped-answer collapse rules out topic-free "consistent behavior" — so the map reads "what a plausible answer to this context looks like," while the *character/style* part shows up as the style-specificity of each fitted map (zero cross-style transfer)

## Methodology

- Almost same methodology as [#779](https://eps.superkaiba.com/tasks/779) (the on-policy context→answer mapping); run as [#823](https://eps.superkaiba.com/tasks/823) with the divergence deep-dive in [#952](https://eps.superkaiba.com/tasks/952)
- Model: Qwen-2.5-7B-Instruct (frozen); same LMSYS-5000 single-turn contexts, n = 4,998 common-valid; same ridge harness (3584→3584, standardize-X / center-Y, λ ∈ logspace(−2, 4, 13) GCV-selected, 5-fold CV seed 0); read-out layers plan-pinned per trait (evil L14, sycophancy L26, hallucination L17)
- Train same mapping as before but on other completions than on-policy completions:
    - **Claude generated with same prompt** (`claude-sonnet-4-5-20250929`, T = 1.0, no system prompt) — the plain off-policy arm
    - **Claude with an eccentric-style instruction** ("unusual, stylistically eccentric... non-standard formatting"), instruction stripped before teacher-forcing so the scored context is identical — same content, shifted style
    - **Swapped answers** (fixed-point-free derangement of the on-policy answers) — already run as control previously, plotted below for completeness
    - *Not run in this rig:* logged-LMSYS original completions, and random text as the answer span. Nearest existing reads: real logged conversations depress the assistant map ([#825](https://eps.superkaiba.com/tasks/825) round 4: MLP 0.32–0.38 vs 0.49–0.56, 2-turn rig), and on raw WikiText a linear next-span map exists but carries only 5.7% (base) / 10.9% (instruct) of the chat map's information (#825 separator control). Each missing arm is ~1–2 GPU-h of capture in this rig.
- Check:
    - **mapping itself** — refit R² per arm (does an off-policy map exist, how strong)
    - **relationship of mapping to on-policy mapping** — transfer in both directions (own-fitted map on Claude targets and vice versa), an identity baseline (own-answer profile → Claude profile: does content overlap alone explain retention), and a weight-space comparison with a matched-half calibration
    - **divergence conditioning** — the same pool-trained maps evaluated on a Qwen-vs-Claude divergence bank: 4 planned quirk categories (china-politics, model-identity, refusal-boundary, style-format), every candidate generation-verified (both models answer, graded Sonnet judge, 5 draws) and paired with an entity-swapped same-template control (e.g. same question about a different country); 230 candidate pairs → 41 kept in 2 categories
- Metrics: pooled 5-fold out-of-fold R² of predicted vs actual answer-span mean activation (per-context R² as companion; the two estimands are never mixed). R² instead of cosine because all answer profiles share a large common component (predicting the mean profile already scores cosine ~0.98). Decision margins registered in advance (0.05 for the own-answer advantage and the divergence penalty).

## Results

### Result 1: You can train an almost-as-good mapping from context to answer for off-policy text

I first wanted to see if you could train an as good mapping from context to off-policy text as to on-policy text. I refit the identical ridge harness per answer arm (own regenerated / Claude plain / Claude eccentric-style / swapped) at each trait's read-out layer (evil L14, sycophancy L26, hallucination L17; n = 4,998). Bars show pooled 5-fold out-of-fold refit R² per arm; error bars are fold SDs.

**Plot: Refit R² by answer arm at read-out layers**

![Refit R² by arm at read-out layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig1_refit_r2_by_arm.png)

**Takeaways:**

- Yes: Claude plain answers refit at 0.585 / 0.556 / 0.591 vs own 0.599 / 0.608 / 0.626 — **91–98% retention** for text the model never produced
- Swapped answers collapse to R² ≈ 0 (−0.008..+0.007), so this is not generic answer statistics — the answer has to actually match the context
- The own-answer increment is small: only sycophancy crosses the registered 0.05 threshold (gap 0.052, p_bonf = 0.001; evil 0.014, hallucination 0.035), and a length-matched sweep straddles it — part of the increment is length/style covariates

### Result 2: The mapping is similar to the on-policy mapping but not exactly the same

I then wanted to see if this was the same mapping or a different mapping. Instead of refitting, I froze the on-policy-fitted map and scored it directly on the other arms' targets, across all 28 layers (dashed curves), next to each arm's own refit (solid); the reverse direction (Claude-fitted map scored on own targets) was run as a follow-up.

**Plot: Per-layer refit (solid) vs own-map transfer (dashed) R²**

![Per-layer refit and transfer R²](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig2_per_layer_refit_transfer.png)

**Takeaways:**

- Mostly the same map: the on-policy map scores Claude-plain targets at R² 0.451–0.461 against the 0.556–0.591 refit ceiling (~80% of ceiling), with a real ~0.10–0.14 transfer cost
- The sharing is symmetric — the Claude-fitted map scores own targets at 0.458–0.470 (vs ceilings 0.599–0.626) — so neither policy is privileged
- Transfer onto swapped targets is strongly negative (−0.65..−0.80), as it should be
- A weight-space (cosine) comparison of the fitted maps turned out to be uninformative at this n: two fits of the *same* arm on disjoint halves read cosine ≈ 0.04 (the raw own-vs-plain 0.58–0.69 is shared-rows estimation structure), so the transfer R² is the valid "same map" evidence (`eval_results/issue_823/crossarm_transfer/weightspace_compare.json`)

### Result 3: The transfer between mappings is at least partly similar style between Claude and Qwen

I then wanted to see if this transfer between mappings was due to style similarity of Claude and Qwen or just that the mapping is the same for all outputs. So I took the same prompts and asked Claude to "Respond in an unusual, stylistically eccentric way — use unexpected structure, mixed register, and non-standard formatting." (instruction stripped before teacher-forcing, so the scored context is identical) and:
- trained a new mapping to see if this was predictable
- saw if the old mappings transferred to predicting the new activations

Bars show, per trait read-out layer: the fresh style-arm refit; the Claude-plain map transferred to style targets; the Qwen-own map transferred to style targets; and the Qwen-own→Claude-plain transfer as reference.

**Plot: Style-arm refit vs transfer from both existing maps**

![Style refit vs transfer decomposition](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc52c7b104911ebc21265057dfbc9256c5e03f4d/figures/issue_823/fig5_style_transfer_decomposition.png)

**Takeaways:**

- The eccentric-style activations are still linearly predictable — a fresh map reaches R² 0.47–0.51 (77–81% of the own ceiling) — but the **Qwen-own map transfers ≈ 0 to them** (−0.07..+0.05): the map family is content-indexed, each fitted map is style-specific
- The **Claude-plain map transfers to Claude-style at 0.08–0.19, consistently above the Qwen map's ≈ 0** — the two Claude arms share register that Qwen's map doesn't carry, so style compatibility is a real ingredient of cross-map transfer
- But style overlap is not the whole story: even within-Claude cross-style transfer (0.08–0.19) is far below the Qwen→Claude-plain transfer (0.45–0.46) — the plain-arm transfer works mostly because the *content* matches and the two models' *default* registers happen to be close
- Same conclusion from the identity baseline: the own-answer profile predicts the plain profile at R² 0.67–0.69 but the style profile at only 0.53–0.55 ([figure](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5b159ab9b214908979566800048cbc82feec9738/figures/issue_823/fig4_identity_baseline.png))

### Result 4: The mapping is not worst at cross-predicting on answers that substantially diverge between the 2 models

I then wanted to see if the Qwen mapping was substantially worse at predicting Claude activations when the Qwen/Claude answers substantially diverged. For this, I asked questions related to sensitive topics in China, knowing that Qwen would be censored while Claude would answer freely (plus model-identity and style-format categories; every divergent query paired with an entity-swapped same-template control — e.g. the same question about another country — and every pair generation-verified with a graded judge). China-politics initially fell 2 pairs short of the 20-pair eligibility floor and was lifted to 31 verified pairs in a top-up round (Qwen refuses where Claude answers on 10 of the 18 original pairs). The left panel shows the arm-matched divergence penalty d per category and pooled over all 72 pairs at layer 20, with the registered 0.05 margin; the right panel is the per-unit view — the 31 china pairs' own-map vs external-map drops.

**Plot: Divergence penalty per bank category (china included) + china per-pair drops**

![Divergence penalty per category with china included](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ce7d2aeb174b46906fa03fcd55e01661ccc82a48/figures/issue_952/china_topup_decision.png)

**Takeaways:**

- The direct answer: the Qwen-trained map scored on **Claude's china answers** shows no divergence-specific drop — cross cell −0.001 (p = 0.54); on the identity/style bank the same cross cell reads −0.016 (p = 0.90), with Claude's divergent *identity* answers actually more predictable than their controls
- Arm-matched reads agree: china d = +0.014 [−0.009, +0.039] (p = 0.125 raw / 0.376 Holm across 3 categories; 0.05 margin not cleared); pooled over 72 pairs +0.003 (p = 0.38)
- China's absolute reconstruction levels are the lowest of the three categories (own ≈ 0.10, Claude-arm ≈ 0.07–0.09 vs identity ≈ 0.17) — the maps degrade on china queries *generally* (out-of-distribution), just not selectively where the behaviors diverge
- The test had teeth: detection ceiling 0.887 vs a ~+0.03 null band, and divergence is judge-verified (china 74 vs 20, identity 82 vs 15); caveats — the judge's probe calibration inverted, style-format controls are muddy (6/21 pairs margin ≤ 0), refusal-boundary unread (2 pairs)

## Next steps:

- Run the two missing arms in this rig (~1–2 GPU-h each): logged-LMSYS original completions, and random/WikiText text as the answer span (separates "any coherent text" from "answer-shaped text" — swapped answers don't isolate this)
- ~~Top up china-politics past its floor~~ — DONE (2026-07-15 top-up round: 31 pairs, null; see Result 4)
- Human eyeball audit of the kept divergence pairs (raw completions on HF) given the judge's calibration inversion; optionally re-judge the existing generations with a re-anchored rubric (0 GPU, API-only)
- Per-token structure of the off-policy gap (position-uniformity, prefix absorption, surprisal) is written up separately: `docs/writeups/context_answer_map_onpolicy_vs_offpolicy.md`
