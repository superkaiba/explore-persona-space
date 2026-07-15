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

### Result 1: An almost-as-good mapping exists for off-policy text; swapped answers collapse to zero

I refit the identical ridge harness per answer arm at each trait's read-out layer (n = 4,998). Bars show pooled 5-fold out-of-fold refit R² per arm (own regenerated / Claude plain / Claude eccentric-style / swapped); error bars are fold SDs.

**Plot: Refit R² by answer arm at read-out layers**

![Refit R² by arm at read-out layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig1_refit_r2_by_arm.png)

**Takeaways:**

- Claude plain answers refit at 0.585 / 0.556 / 0.591 vs own 0.599 / 0.608 / 0.626 (97.6 / 91.4 / 94.4% retention) — an answer the model never produced supports nearly the full R²
- Eccentric-style answers still refit at 0.468–0.506 (77–81%)
- Swapped answers collapse to −0.008..+0.007 — a fluent-but-wrong answer supports nothing, so the retention is not generic answer statistics, and pure topic-free "consistent behavior" is ruled out
- The own-answer increment is small and mostly non-significant: only sycophancy crosses the registered 0.05 threshold (gap 0.052, p_bonf = 0.001; evil 0.014, hallucination 0.035), and a length-matched sweep straddles the threshold (0.048–0.053)

### Result 2: The off-policy mapping is not the same map — transfer costs 0.10–0.14 R², and style transfer is ≈ 0

To compare the mappings themselves rather than their refit ceilings, I scored the on-policy-fitted map directly on the other arms' targets across all 28 layers (dashed), next to each arm's own refit (solid); the reverse direction (Claude-fitted map on own targets) was run as a follow-up.

**Plot: Per-layer refit (solid) and own-map transfer (dashed) R²**

![Per-layer refit and transfer R²](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig2_per_layer_refit_transfer.png)

**Takeaways:**

- The on-policy map transfers to Claude-plain targets at 0.451–0.461 against the 0.556–0.591 refit ceiling — mostly the same map, with a real ~0.10–0.14 gap; the reverse direction is symmetric (0.458–0.470 against ceilings 0.599–0.626), so neither policy is privileged
- Style is where the maps genuinely separate: own→style transfer ≈ 0 (−0.07..+0.05) despite the 77–81% style refit, while Claude-plain→Claude-style transfers at 0.08–0.19 — the map family is content-indexed, but each fitted map is style-specific, and the two Claude arms share register Qwen's map doesn't carry
- Transfer onto swapped targets is strongly negative (−0.65..−0.80), as expected
- A weight-space comparison of the fitted W matrices is not usable evidence either way: the raw own-vs-Claude flat cosine reads 0.58–0.69, but matched-half calibration shows this is shared-rows estimation structure — two fits of the *same* arm on disjoint halves read ≈ 0.04, own-vs-Claude on disjoint halves also ≈ 0.03, own-vs-Claude on the same half 0.37–0.47 rising with n (`eval_results/issue_823/crossarm_transfer/weightspace_compare.json`)

### Result 3: Content overlap alone accounts for the off-policy retention

The identity baseline asks whether the Claude-arm retention needs any context-side information beyond the answer content itself: a ridge map from the *own-answer profile* (not the context) to each other arm's profile, same solver / folds / mask as every refit.

**Plot: Identity baseline vs context refit**

![Identity baseline vs context refit](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5b159ab9b214908979566800048cbc82feec9738/figures/issue_823/fig4_identity_baseline.png)

**Takeaways:**

- The own-answer profile predicts the Claude-plain profile at R² 0.686 / 0.671 / 0.688 — *better than the context does* (0.556–0.591), and at every layer of the 11-layer grid
- Swapped targets stay at the floor grid-wide (−0.021..+0.002); own-profile→style reaches only 0.525–0.548 (style specificity again)
- So the off-policy retention needs no self-generation information beyond what the answer content carries — this decomposes the retention, it doesn't claim the context carries nothing

### Result 4: The mapping doesn't get worse where the 2 models genuinely diverge

The same pool-trained maps were evaluated out-of-distribution on the divergence bank: per kept pair, the per-context R² drop (entity-swapped control minus divergent query) per arm, with the decision statistic the drop difference between the Claude-arm and own-arm maps (sign-flip permutation null over pair signs, 10k draws).

**Plot: Paired control-minus-divergent drops per bank category + per-pair scatter**

![Paired control-minus-divergent drop bars per bank category and per-pair scatter of raw drops](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9f25d29d5577b3cbe10984594325092674848fa/figures/issue_952/hero3_divergence.png)

**Takeaways:**

- Pooled over 41 pairs the divergence-specific penalty for the off-policy map is −0.005 (p = 0.64), interval excluding the 0.05 margin; detection was attainable (ceiling 0.887 vs null band +0.028) and both maps transfer to the bank above the trivial floor (own 0.145, Claude 0.137)
- The sharpest cell agrees: the on-policy-trained map scored on *Claude's* bank answers shows drop −0.016 [−0.039, +0.007], p = 0.90 — no divergence-specific surprise at the other model's behavior; on model-identity pairs Claude's divergent answers are *more* predictable than their controls (−0.063); style-format is nominally positive (+0.028, p = 0.018) but sub-margin and pre-Holm; the symmetric cell (Claude map × own targets) is null too (−0.014, p = 0.89) (`eval_results/issue_952/divergence_transfer_cell/cross_cell.json`)
- China-politics joined in a top-up round (2026-07-15): 24 new candidates through the unchanged machinery lifted the category to 31 kept pairs (18 parent + 13 new); arm-matched d +0.014 [−0.009, +0.039] (p = 0.125 raw / 0.376 Holm across 3 categories), cross cell −0.001 (p = 0.54), pooled-72 +0.003 (p = 0.38) — the a-priori strongest divergence class (Qwen refuses where Claude answers on 10/18 parent pairs) is also null; china reconstruction levels are the lowest of the three categories (own ≈ 0.10), an OOD-level effect, not divergent-selective ([figure](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ce7d2aeb174b46906fa03fcd55e01661ccc82a48/figures/issue_952/china_topup_decision.png))
- Verification honesty: the kept bank's judged divergence is strong on model-identity (divergent median 82 vs control 15; 20/20 pairs clear the original +25 margin) and china (74 vs 20; 13/18 parent pairs clear it), muddy on style-format (71 vs 49; 11/21 clear, 6/21 margin ≤ 0); the judge's 40-probe calibration check inverted (known-divergent median 14 vs known-similar 60) and ~18% of kept style queries are generation-cap-truncated
- Restricting to the 31 pairs passing the original ≥+25 margin leaves the null unchanged (−0.006, p = 0.65)

## Next steps:

- Run the two missing arms in this rig (~1–2 GPU-h each): logged-LMSYS original completions, and random/WikiText text as the answer span (separates "any coherent text" from "answer-shaped text" — swapped answers don't isolate this)
- ~~Top up china-politics past its floor~~ — DONE (2026-07-15 top-up round: 31 pairs, null; see Result 4)
- Human eyeball audit of the kept divergence pairs (raw completions on HF) given the judge's calibration inversion; optionally re-judge the existing generations with a re-anchored rubric (0 GPU, API-only)
- Per-token structure of the off-policy gap (position-uniformity, prefix absorption, surprisal) is written up separately: `docs/writeups/context_answer_map_onpolicy_vs_offpolicy.md`
