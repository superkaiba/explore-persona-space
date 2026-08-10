# Steering and patching at the context vector: causal effects on the answer

Writeup of [#1415](https://eps.superkaiba.com/tasks/1415), [#2094](https://eps.superkaiba.com/tasks/2094), [#2162](https://eps.superkaiba.com/tasks/2162) (2026-08-10). All numbers below are from on-policy generations of Qwen-2.5-7B-Instruct (greedy one-draw grids plus temperature-1.0 anchor/confirmation draws, 5–10 per cell), judged by claude-sonnet-4-5 graded 0–100; no teacher forcing on any headline read.

## Motivation

We have a linear mapping from the context vector $v_C$ (last-prompt-token residual state, at the newline before the assistant header) to the answer vector $V_a$ (mean residual state over the model's own answer). It predicts the answer state well when read passively, suggesting a lot of the context's information is concentrated at that single position. If so, intervening **only at the context vector** (or only at the prefix-end state — the last token of the prefix, before the final user query) should causally move the entire answer, at both the activation and the behavior level. We tested this.

## Methodology

We take the previously trained linear mappings as frozen prediction instruments ([#779](https://eps.superkaiba.com/tasks/779)'s 963k-context ridge maps at context-end L14/L19, [#1738](https://eps.superkaiba.com/tasks/1738)'s prefix-end maps, [#922](https://eps.superkaiba.com/tasks/922)'s L20 map) — no new map training.

Three settings from one crossed bank (#2094: 3 prefixes × 5 queries → 60 pairs):

- **Matched query** — different prefix, same query (can the intervention change the "interpretation"/register of one query?)
- **Matched prefix** — same prefix, different query (can it change which query the model answers?)
- **Cross** — different prefix AND different query (both at once)

Prefixes/queries span persona prompts, behavioral instructions, and a constructed multi-turn conversation (#2094, #1415); the information-type sweep #2162 extends this to 21 minimal-pair types including ICL examples and real WildChat prefixes. Full slot × layer heatmaps + qualitative examples: [docs/issue2094_patching_dashboard.md](https://github.com/superkaiba/explore-persona-space/blob/main/docs/issue2094_patching_dashboard.md).

**Interventions:** add α·(v_C(B) − v_C(A)) at one position, α ∈ {0.5, 1, 2, 4}, or replace the full state (= activation patching), at each of the 28 layers singly, jointly at the middle layers 14–20 (where the mapping works best), and jointly at all layers. Slots: context-end, prefix-end, 2nd/3rd-to-last tokens, last-3 jointly, query span; plus a prefix-centroid direction (mean of v_C over queries) at context-end. Every cell has a norm-matched shuffled-donor null (wrong-pair donor).

**Metrics** (per pair; floor = unpatched under context A, ceiling = generate under context B):

- Behavior: two judge calls per draw ("does this express A?" / "…B?"), Δ = (judge_B − judge_A)/100, and **F = (Δ_patched − Δ_floor)/(Δ_ceiling − Δ_floor)** — the fraction of a full context swap the patch recovers.
- Activation: **F_act** = signed projection of the answer-vector shift onto the floor→ceiling axis (same normalization, disjoint baseline halves); plus cosine of the realized shift to the target shift, and the transport cosine to the mapping's predicted shift f(v_C + Δ) − f(v_C).

## Result 1: steering at the context vector moves the answer vector — but not linearly, and not as the mapping predicts

Steering with Δ = v_C(B) − v_C(A) at the single last-context token moves the realized answer state toward the target: cosine 0.36–0.41 to the target shift after shared-baseline correction, 28/28 pairs above their random-direction nulls, peaking at the middle layers 14–17 — but the traversal covers only 2–5% of the target-shift norm (#1415). On the fraction-of-swap scale, the strongest clean activation movement is F_act 0.41 (vs null 0.09) at the all-layer full-state patch and 0.27 (vs 0.03) at the mid-stack joint edit at α = 0.5, matched-query at context-end. Prefix-end moves more in raw norm but is not pair-specific (best cell: 0.210 steered vs 0.209 null).

![Per-layer geometric alignment and judged behavior shift, both peaking mid-stack](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7d4f686bfb519175780c9e789d827b8610bce160/figures/issue_1415/layer_profile_geometry_vs_behavior.png)

![Activation fraction-of-swap heatmap across slots, layers, doses, settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/hero1_f_act_heatmap.png)

**The mapping does not predict the steered shift.** The transport cosine between the map-predicted and realized shift is 0.00 at L20 (#1415; magnitude over-predicted ~16×) and tops out at 0.16 at context-end L14 in #2094 (pooled means ≤ 0.09).

![Banked-map transport cosines, steered vs null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/result1b_transport_cosines.png)

**The steering response is not linear either:** shift magnitude is nearly flat over the 8× dose range (log-log slope 0.00–0.06 vs 1.0 for a linear response), the shift direction drifts across doses (adjacent-dose cosine 0.62 at L14, despite split-half reliability 0.93 — real drift, not noise), and the best one-operator linear fit of edit → response explains ≤ 8.4% of held-out variance. Coherence never limits single-position edits (0.7% incoherent vs 2.0% unpatched baseline), so the dose ladder was never cut off by incoherence — the response simply saturates. Only multi-token mid-stack edits (query span, last-3) break fluency, and they break it for the null too (generic disruption).

## Result 2: behavior follows — only at the context vector, only partially, and mostly for the prefix/persona

Grid-wide over 1,245 behavior cell families (well-separated anchor pairs only), 15 families separate cleanly from the shuffled-donor null — **every one at context-end**: 14 matched-query (prefix/persona transfer) + 1 cross. Prefix-end, 2nd-to-last, and 3rd-to-last give zero null-separated families anywhere. Sizes: single layers 12–20 read F 0.18–0.33, the mid-stack joint edit 0.43 at α = 0.5, and the all-layer full-state patch 0.63 (0.51 under independent temperature-1.0 re-sampling; null 0.10) — the maximal single-position patch recovers roughly half to two-thirds of a full context swap, never all of it. 10 of the 15 families confirm on fully disjoint CIs under re-sampling. #1415's independent bank agrees: judged behavior peaks exactly where the geometric alignment peaks (layer 14: +6.2 judge points = 21% of the context-swap ceiling, p = 0.008, replicated on two fresh seeds), matched-query only.

Steering strength buys almost nothing: F does not rise monotonically from α = 0.5 to 4 (the dose curves are flat), consistent with the saturating activation response above. Note the α = 1 single-layer difference-add is equivalent to patching that one layer; the "replace" arm is the full activation patch.

![Behavior fraction-of-swap heatmap across slots, layers, doses, settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/hero2_f_beh_heatmap.png)

![Behavior F dose response at the banked layers 14/19, steered vs shuffled-donor null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/267387d2c9b448d3d0f29a1cb7bc8e43cd222e36/figures/issue_2094/result_fbeh_dose_L14L19_wellsep.png)

![Forest plot: steered vs null for the 15 surviving families under re-sampling](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99448919d1972edd45bd12426dbc57c3fb4ff9e8/figures/issue_2094/fu1_conf1_forest.png)

## Result 3 (+4): query recognition — the context vector does not carry which query is answered; the query's own token states do

Matched prefix (same prefix, different query): patching the context vector produces **zero query transfer** — the selected cells confirm at F ≤ 0.02 and the grid-wide screen has no matched-prefix survivor; #2162's `query_content` cell replicates the null. Patching the whole query span including chat-template tokens is untestable — it destroys fluency in both arms.

What does work: replacing the query's **text-token states only** (template tokens excluded), at all layers or the mid-stack band — the model fluently answers the target query in the source prefix's register (e.g. pirate prefix + sky-color query patched toward rent-vs-buy → a pirate-register rent-vs-buy answer). 8 clean query-rubric reads at F 0.68–0.95, against nulls elevated to 0.22–0.56 (with a 5-query bank, a shuffled donor often installs the same target query), all at α = 0.5 or the mid-stack joint band.

Cross (prefix and query both different, the old Result 4) barely transfers: 1 clean context-end family, and cross steering vectors transport worse than matched-query ones (cosine 0.22 vs 0.49, #1415).

![The 14 clean query-text reads, steered vs null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/493ed29a255022bd0c5bec107bb3a71796837a70/figures/issue_2094/fu2_qtext_clean_forest.png)

## Which kinds of information are causally usable at the context vector (#2162)

A patch-only sweep over 21 minimal-pair information types (facts, stated vs demonstrated policies, ICL task definitions, personas, discourse state; WildChat carriers) at both slots, with a linear read probe alongside: the probe decodes almost everything from the natural states (55 of 76 cell × slot combos are stored-but-unusable), but the patch moves behavior for only **5 cells — all the stated-formatting-policy family at context-end** (`instr_format` plus its load and conflict variants, F 0.70–0.81 vs nulls ≈ 0.07–0.14). Retrievable facts (user's name, etc.), ICL task mappings, and demonstrated (as opposed to stated) policies are decodable but not causally usable through the single-position patch.

![Read × write 2×2: probe AUC vs causal F per type](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/two_by_two.png)

## Takeaways

1. The context vector is causally special but not causally sufficient: it is the only single position where activation edits move behavior past the null, yet even the maximal all-layer patch recovers only ~0.5–0.65 of a full context swap, and what transfers is the prefix's persona/register and stated formatting policy — not query identity, not stored facts.
2. Passive predictability ≠ causal control: the fitted linear map does not transport to interventions (transport cosine ≤ 0.16), and the edit → response relation is far from linear (flat in dose, direction drifts).
3. Query identity lives in the query's own token states: replacing those (template-stripped) swaps which query is answered at up to F ≈ 0.95 while the prefix register survives — the two channels are separable.

**Sources:** clean-result bodies of #1415 / #2094 / #2162; slot × layer heatmap dashboard `docs/issue2094_patching_dashboard.md`.
