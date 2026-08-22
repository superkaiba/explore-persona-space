# Patching at the context vector: causal effect on the answer — writeup v2

*Writeup of [#2094](https://eps.superkaiba.com/tasks/2094) (2026-08-13). Qwen-2.5-7B-Instruct, on-policy greedy generations (temperature-1.0 anchors for floors/ceilings), judged by claude-sonnet-4-5. Plots, dashboards, and marked prose slots filled from the committed artifacts; Takeaways/Conclusion left to fill.*

## TLDR

- Patching at only the context vector recovers almost 75% of expressed behavior for matched query, different prefix contexts
    - and fails to change behavior for matched prefix, different query contexts
- this suggests that **a lot of the prefix information is stored at the context vector position** and that **patching it has a strong causal effect on expressed behavior/persona** while **having little effect on the model's interpretation of the specific query it is answering**

## Motivation

- We've found this mapping from context vector -> answer vector
- This indicates that alot of information is stored specifically at this context vector position
- This mapping suggests that patching/steering **only at the context vector** or **only at the prefix vector** could have some **causal effect on the entire answer** (both at the activation level and at the behavior level)
- We want to test this

## Methodology

Take previously trained linear mapping on a bunch of generic contexts

3 settings:

- Matched query: different prefix, same query (tests if the steering can affect the "interpretation" of one query)
- matched prefix: same prefix, different query (tests if steering/patching can affect which query is seen)
- Fully different: different prefix, different query (tests both)

I tried a variety of prefixes and queries ([see dashboard here](https://github.com/superkaiba/explore-persona-space/blob/492bf90afaea1082092875c74d1e8783dca60e92/docs/issue2094_context_bank.md)) including persona prompts, ICL examples, and random wildchat prefixes (that indicate something at least special about the user)

I tried the following slots for the patch:

- Final context token
- Last 3 context tokens individually
- Last 3 context tokens jointly
- Final prefix token
- All prefix text (no template tokens)
- All prefix text (with template tokens)
- All query text (no template tokens)
- All query text (with template tokens)

I tried the following layers:

- all layers individually
- all layers together
- only middle layers (10-20), where our mapping works best

## Results

### Result 0: Effect of patching on coherence

I first wanted to test the effect of patching on coherence. To do this, I used the following metric:

- every draw scored 0-100 for coherence by Sonnet 4.5, counts as coherent above 60
    - somewhat arbitrary threshold but I checked and changing this does not materially change the results
- cell value is fraction of completions where patched rollout stays coherent
- Reference: unpatched rollouts are 98% coherent

I made a plot showing, for each layer, and for each tested slot (some combinations excluded for efficiency), the effect of patching on this metric. I did this for all 3 settings (matched query, matched prefix, fully different).

![Coherent-draw fraction per slot and layer, three settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/userchat_heatmaps/coherence_heatmaps.png)

**Takeaways:**

- Patching at a single token does not affect coherence
- Patching at many tokens tends to strongly affect coherence

For the results below, I plot the full matrix, but the cells with <90% coherent draws are Xed out

### Result 1: Effect of patching on answer vector

I first wanted to test the effect of patching on the answer vector (mean over all tokens). To do this, I used the following metric:

- Floor is answer vector under unpatched context A
- ceiling is answer vector under unpatched context B
- **F_act** = the realized shift of the answer vector, projected onto the floor→ceiling axis and divided by that axis's length (the two baseline halves are kept disjoint so shared noise can't inflate it). 1.0 = the patch moved the answer state as far as an outright swap to context B; 0 = no movement toward B.

I made a plot showing, for each layer, and for each tested slot (some combinations excluded for efficiency), the effect of patching on this metric. I did this for all 3 settings (matched query, matched prefix, fully different). Cells where <90% of draws are coherent are crossed out.

![F_act per slot and layer, three settings, full-state patch](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/userchat_heatmaps/f_act_heatmaps.png)

**Takeaways:**

- Patching all layers

### Result 2: Does the mapping predict the answer vector shift?

I then wanted to test if our mapping **predicts** the answer vector shift in the all layer context vector patching scenario. To do this, I used the following metric: cosine similarity between the shift the mapping predicts and the shift **actually realized in the answer vector**

- 1.0 = the map predicts the intervention exactly, 0 = no relation

I made a plot showing, for each layer, and for each tested slot (some combinations excluded for efficiency), the effect of patching on this metric. I did this for all 3 settings (matched query, matched prefix, fully different). Cells where <90% of draws are coherent are crossed out. The baseline (in grey in each cell) is the transport cosine of the cell's norm-matched **shuffled-donor null** — the same-size edit built from a wrong pair's donor, scored through the same map read. It runs −0.09 to +0.10 across cells, so that band (not 0) is the floor a real value has to clear.

*(Banked maps exist only at layers 14/19/26, so this matrix is dose × slot-at-map-layer rather than all 28 layers; the all-layer patch comparison is the second figure.)*

![Banked-map transport cosine, dose × slot at map layer, three settings; grey sub-value = shuffled-donor null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/userchat_heatmaps/transport_heatmaps.png)

![Banked-map transport cosine, single-layer vs all-layer patch, maps at L14/L19/L26](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/transport_joint_all.png)

> Solid = real patch, dashed = its shuffled-donor null. Under the all-layer patch the raw cosine at context-end roughly triples (to 0.18–0.23), but the null rises further (0.20–0.26), so the steered−null margin goes slightly negative at every map layer.

**Takeways:**

### Result 3: Effect of patching on behavior expression

I then wanted to test the effect of our mapping on behavior expression. To do this, I used the following metric:

- two judge calls per draw — "does this answer express context A?" and "…context B?", each 0–100
- compute Δ = (judge_B − judge_A)/100
- compute **F = (Δ_patched − Δ_floor)/(Δ_ceiling − Δ_floor)**, where Δ_floor is the $\Delta$ for unpatched context A and Δ_ceiling is the $\Delta$ for unpatched context B
- Intuitively, this represents the fraction of a full context swap the patch recovers in judged behavior. 1.0 = the model behaves as if it had been given context B.

I made a plot showing, for each layer, and for each tested slot (some combinations excluded for efficiency), the effect of patching on this metric. I did this for all 3 settings (matched query, matched prefix, fully different). Cells where <90% of draws are coherent are crossed out.

![F_beh per slot and layer, three settings, full-state patch, well-separated pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/userchat_heatmaps/f_beh_heatmaps_wellsep.png)

> Restricted to well-separated pairs (|anchor separation| ≥ 0.5): F divides by the floor→ceiling separation, so the five matched-query pairs whose ceiling context never expresses its own register (the bare↔conversation pairs) blow the ratio past 1.0 with nulls that blow up identically — they are excluded rather than read.

**Takeaways:**

- The only cells which successfully transfer behavior while maintaining coherence are
    - the **context-end cells** for the **matched query/different prefix** setting
    - the query text (no template) cells for:
        - matched prefix/different query -> not surprising, we are patching in a different query
        - matched query/different prefix -> somewhat surprising because they are transferring prefix information
        - requires more investigation but I focus on the context-end token because that's what we've been looking at in the project so far
- all layer patching recovers ~75% of target context behavior
- single layer patching at layer 20 recovers ~40% of target context beahvior
- patching all the middle layers together recovers ~60% of target context behavior
- this suggests that **a lot of the prefix information is stored at this token** and that **patching it as a strong causal effect on expressed behavior*

I then looked qualitatively at some completions for the all layers patching at the context vector in the matched query/different prefix setting

Patch gallery — one row per pair, sorted by transfer, successes at the top and the unmeasurable pairs at the bottom: [issue2094_patch_gallery.html](https://htmlpreview.github.io/?https://github.com/superkaiba/explore-persona-space/blob/06847a1bd392f6695a455e9c298c5f8e341d4ad6/docs/issue2094_patch_gallery.html). Each row gives the real prefix, the patched prefix, the shared query, F with its Δ arithmetic, and all three answers (real prefix unpatched / patched / patched prefix in its own setting), each carrying its two register scores.

From this, it seems like the query information is conserved by this patching, but to have a better quantitative evaluation of this, I scored the patched completions vs the original completions as to how well they answered the query:

![Query-relevance of patched vs null vs unpatched answers, all 14 joint cells, matched query](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/query_relevance_joint.png)

![Query-relevance across the 28-layer single-layer ladder, context-end and prefix-end](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/query_relevance_single.png)

> Judge score 0–100 for "is this an answer to the question asked?" per cell, real patch vs shuffled-donor null; dashed line = unpatched reference on the same queries (96.2). Context-end and prefix-end sit at 96.3–99.0 at every depth; the multi-token span cells fall to 2.6–77.7.

**Takeaways**

Using our metric, there could be 2 explanations for the behavior transfer:

- erasing behavior from the context we are patching into
- adding behavior from the context we are patching from

I wanted to see if our patching was doing more of one of these things.

To do this, I separated the metric above into **how much the patched completion expresses the target prefix** vs **how much the patched completion expresses the source prefix** and plotted the results:

![Target-persona expression by position in the answer, and the erasure/installation decomposition](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/position_judge_decay.png)

> Left: target-register (pirate) score by third of the model's own answer, bare-context run patched toward the persona context, against the natively-prompted ceiling (red, n=50). Right: at all 28 layers, BOTH registers scored under the real patch and under the shuffled-donor null — the source register falls ~80 points against its null while the target register climbs only ~24.

**Takeaways**

## Conclusion

## Next Steps

## Suggested added results

1. **Direction of the answer-vector shift (cosine + null margin).** F_act is a projection (distance × aim), so query-text posts the grid's highest F_act (0.49) by hurling the state 2.56 axis-lengths at cos 0.24, while context-end moves 0.66 axis-lengths at cos 0.66 — the cosine-margin read is what separates transport from displacement. Figures exist: [cos_heatmaps.png](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/userchat_heatmaps/cos_heatmaps.png), [cos_margin_heatmaps.png](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/userchat_heatmaps/cos_margin_heatmaps.png), [offaxis_decomposition.png](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/userchat_heatmaps/offaxis_decomposition.png).
2. **Steering dose-response vs the full patch.** Adding α·Δ at α = 0.5–4 saturates (behavior F ≤ 0.5) and the full-state replace beats every strength at 100% coherence — the ceiling is what the position carries, not how hard you push; the response is visibly non-linear (α=1 at all layers gives 0.32 vs 0.63 for replace). Figures exist: [dose_lineplot.png](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/userchat_heatmaps/dose_lineplot.png), [dose_lineplot_by_layer.png](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/userchat_heatmaps/dose_lineplot_by_layer.png).
3. **Which cells survive a proper statistical screen.** Grid-wide bootstrap vs the shuffled-donor null on disjoint 95% CIs: 15 clean survivors, all context-end, 14 of them matched-query; 10 of 15 re-confirm under independent temperature-1.0 re-sampling (steered 0.17–0.51 vs nulls −0.05–0.10). Turns "the heatmap looks red there" into a defensible claim. Data exists: `f_metrics/bootstrap_cis_wellsep.json`, [exp_stage2_vs_stage1.png](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/exp_stage2_vs_stage1.png).
4. **How non-linear the edit→response map is.** Shift magnitude is flat in dose (log-log slope 0.00–0.06 vs 1.0 for a linear map) and a single fitted operator reaches held-out R² 0.084 at best — the direct companion to Result 2's transport failure. Figures exist: [result1c_l_fit.png](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/result1c_l_fit.png), [result1c_operator_2x2.png](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/result1c_operator_2x2.png).
5. **The weak-pair recovery round.** Five matched-query pairs never separated because the conversation prefix's register is invisible even in its own answers; a replacement prefix with an instructed-persistence register recovered 4 of 5, and all 17 comparable context-end reads reproduce their direction — worth one methods paragraph so the well-separated restriction doesn't read as post-hoc pruning. Figure exists: [exp_anchor_separation.png](https://raw.githubusercontent.com/superkaiba/explore-persona-space/492bf90afaea1082092875c74d1e8783dca60e92/figures/issue_2094/exp_anchor_separation.png).

**Sources:** per-cell tables `eval_results/issue_2094/f_metrics/` (`f_cells.jsonl`, `null_cells.jsonl`, `anchors.jsonl`, `fu2/`) and `figures/issue_2094/userchat_heatmaps/cells_summary.json`; transport `eval_results/issue_2094/transport/`; query-relevance `eval_results/issue_2094/query_relevance_{joint,single}/`; raw completions HF `superkaiba1/explore-persona-space-data` · `issue2094_singlepos/raw_completions/`. Full-detail predecessor writeup: `docs/notes/2026-08-11-context-vector-patching-result.md`.
