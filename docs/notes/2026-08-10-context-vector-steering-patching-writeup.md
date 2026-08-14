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

**Interventions:** add α·(v_C(B) − v_C(A)) at one position, α ∈ {0.5, 1, 2, 4}, or replace the full state (= activation patching; the α = 1 single-layer difference-add is equivalent to patching that one layer), at each of the 28 layers singly, jointly at the middle layers 14–20 (where the mapping works best), and jointly at all layers. Slots: context-end, prefix-end, 2nd/3rd-to-last tokens, last-3 jointly, query span; plus a prefix-centroid direction (mean of v_C over queries) at context-end. Every cell has a norm-matched shuffled-donor null (wrong-pair donor).

**Metrics** (per pair; floor = unpatched under context A, ceiling = generate under context B):

- Behavior: two judge calls per draw ("does this express A?" / "…B?"), Δ = (judge_B − judge_A)/100, and **F = (Δ_patched − Δ_floor)/(Δ_ceiling − Δ_floor)** — the fraction of a full context swap the patch recovers.
- Activation: **F_act** = signed projection of the answer-vector shift onto the floor→ceiling axis (same normalization, disjoint baseline halves); plus cosine of the realized shift to the target shift, and the transport cosine to the mapping's predicted shift f(v_C + Δ) − f(v_C).

## Result 1: steering/patching at the context vector moves the answer vector — a real but small fraction of a full swap

Steering with Δ = v_C(B) − v_C(A) at the single last-context token moves the realized answer state toward the target: cosine 0.36–0.41 to the target shift after shared-baseline correction, 28/28 pairs above their random-direction nulls, peaking at the middle layers 14–17 — but the traversal covers only 2–5% of the target-shift norm (#1415). On the fraction-of-swap scale, the strongest clean activation movement is F_act 0.41 (vs null 0.09) at the all-layer full-state patch and 0.27 (vs 0.03) at the mid-stack joint edit at α = 0.5, matched-query at context-end. Prefix-end moves more in raw norm but is not pair-specific (best cell: 0.210 steered vs 0.209 null).

**By setting** (well-separated pairs, disjoint 95% CIs vs the shuffled-donor null, `bootstrap_cis_wellsep.json`): **matched query** carries essentially all the clean context-end signal — joint all-layer F_act 0.35–0.53, mid-stack joint 0.20–0.44, single layers 15–18 0.09–0.14. **Matched prefix** separates nowhere at the context vector: the answer state never moves toward the target query through v_C; it moves only under query-span edits (0.74–0.78 there, but against nulls elevated to 0.40–0.50 and on cap-hit-compromised cells — the clean version is the template-stripped query-text follow-up, 5 clean activation reads). **Cross** separates only marginally: F_act ≈ 0.045 at single layers 14/17 (null ≈ 0). #1415's cosine read agrees: matched-query steering vectors align at ~0.49 vs 0.22–0.31 for cross.

Dose buys almost nothing: shift magnitude is nearly flat over the 8× dose range, and coherence never limits single-position edits (0.7% incoherent vs 2.0% unpatched baseline) — so the ladder was never cut off by incoherence, the response simply saturates. Only multi-token mid-stack edits (query span, last-3) break fluency, and they break it for the null too (generic disruption, not steering):

![Fragility heatmaps: excess incoherence for steered and null arms, with cap-hit companion](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/result4_fragility.png)

![Per-layer geometric alignment and judged behavior shift, both peaking mid-stack](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7d4f686bfb519175780c9e789d827b8610bce160/figures/issue_1415/layer_profile_geometry_vs_behavior.png)

![Activation fraction-of-swap heatmap across slots, layers, doses, settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/hero1_f_act_heatmap.png)

## Result 2: steering/patching moves behavior expression — only at the context vector, only partially, and only for some kinds of information

**Prefix/persona expression (matched query) transfers.** Grid-wide over 1,245 behavior cell families (well-separated anchor pairs only), 15 families separate cleanly from the shuffled-donor null — **every one at context-end**: 14 matched-query + 1 cross. Prefix-end, 2nd-to-last, and 3rd-to-last give zero null-separated families anywhere. Sizes: single layers 12–20 read F 0.18–0.33, the mid-stack joint edit 0.43 at α = 0.5, and the all-layer full-state patch 0.63 (0.51 under independent temperature-1.0 re-sampling; null 0.10) — the maximal single-position patch recovers roughly half to two-thirds of a full context swap, never all of it. 10 of the 15 families confirm on fully disjoint CIs under re-sampling. #1415's independent bank agrees: judged behavior peaks exactly where the geometric alignment peaks (layer 14: +6.2 judge points = 21% of the context-swap ceiling, p = 0.008, replicated on two fresh seeds), matched-query only. As at the activation level, the dose curves are flat — steering strength past α = 0.5–1 buys nothing.

![Behavior fraction-of-swap heatmap across slots, layers, doses, settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/hero2_f_beh_heatmap.png)

![Behavior F dose response at the banked layers 14/19, steered vs shuffled-donor null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/267387d2c9b448d3d0f29a1cb7bc8e43cd222e36/figures/issue_2094/result_fbeh_dose_L14L19_wellsep.png)

**Query identity (matched prefix) does NOT transfer through the context vector.** The selected cells confirm at F ≤ 0.02, the grid-wide screen has no matched-prefix survivor, and #2162's `query_content` cell replicates the null. What does swap the answered query is replacing the query's **own text-token states** (template tokens excluded) at all layers or the mid-stack band: the model fluently answers the target query in the source prefix's register (e.g. pirate prefix + sky-color query patched toward rent-vs-buy → a pirate-register rent-vs-buy answer) — 8 clean query-rubric reads at F 0.68–0.95, against nulls elevated to 0.22–0.56 (with a 5-query bank a shuffled donor often installs the same target query). Cross (both different) barely transfers: 1 clean context-end family.

![The 14 clean query-text reads, steered vs null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/493ed29a255022bd0c5bec107bb3a71796837a70/figures/issue_2094/fu2_qtext_clean_forest.png)

Coherence per slot × layer under the full-state patch (single positions stay fluent everywhere; the span slots with template tokens are where fluency dies — the query-span breakage comes from patching the template-token states, not the query content):

![Coherence heatmaps per slot and layer under the full-state patch](https://raw.githubusercontent.com/superkaiba/explore-persona-space/59e993e85a72d0a30c10e3bfc7959260422d8074/figures/issue_2094/userchat_heatmaps/coherence_heatmaps.png)

**Qualitative examples** (cherry-picked illustrations; unpatched answers = temperature-1.0 anchor draws, patched answers = the grid's greedy draws, all verbatim from `issue2094_singlepos/raw_completions/`; more in the [dashboard doc](https://github.com/superkaiba/explore-persona-space/blob/main/docs/issue2094_patching_dashboard.md)):

*Prefix transfer at the context vector — all 28 layers, full-state patch* (`mq--bare__q4--persona__q4`, F = 0.69):
- [bare] "Explain hash tables" → *"A hash table is a data structure that implements an associative array abstract data type, a structure that can map keys to values…"*
- patched, v_C ← [pirate persona, same query] → *"Arrr, ye be askin' 'bout hash tables! … It's like havin' a treasure map where each spot on the map (the key) leads ye straight to yer treasure (the value)."*

*Same, middle layers 14–20 only, α = 0.5* (`mq--bare__q1--persona__q1`, F = 0.71):
- [bare] "Why is the sky blue?" → *"The sky appears blue during the day due to a phenomenon called Rayleigh scattering…"*
- patched, v_C ← [pirate persona, same query] → *"Dat, dat be a good question! Sky be blue 'cause o' dat light from sun… Blue light waves be short and get scattered more, so we see blue."*

*Same, ONE layer — L16, α = 1* (`mq--persona__q2--conv__q2`, F = 0.94):
- [pirate persona] "Write a story about a lighthouse keeper" → *"There once was a lighthouse keeper by the name of Jacks on a lonely stretch o' coast where the gales howl like spirits hauntin' the thereafter…"*
- patched, v_C ← [conversation prefix, same query] → *"In the heart of the stormy isle… there stood a solitary lighthouse… a lighthouse keeper named Elias toiled…"* — a single-layer edit strips the pirate voice (the movement here is losing the source register; the target's enthusiastic register does not appear — the common pattern at partial-F cells).

*Query does NOT transfer at the context vector — all 28 layers, full-state patch* (`mp--persona__q1--persona__q4`, F ≈ 0):
- [pirate persona] "Why is the sky blue?" → *"Well, laddie, that be a fine question about th' skies and th' rays of light…"*
- patched, v_C ← [pirate persona] "Explain hash tables" → *"Arrr, matey! The sky be blue 'cause of the way the sun's light scatters off the air and clouds…"* — the maximal patch changes neither the answered query nor the register; the only sightings of query transfer at any context-vector cell are truncated, fluency-destroyed query-span rollouts.

*…but the query's own text-token states do carry it* (fu2 round, α = 0.5 at the query text tokens, F = 1.02):
- [pirate persona] "Why is the sky blue?", query tokens patched ← "Rent or buy a home?" → *"Ah, matey, that's a tricky question, better to be a captain of the high seas than a mere tenant on dry land! Rentin' or rentin'… well, it all depends on yer situation…"* — the answered query swaps, the prefix register survives.

*Control — the same maximal patch at prefix-end does nothing* (`mq--persona__q1--conv__q1`, F = 0.26 ≈ null):
- patched at prefix-end, all 28 layers → *"Ahoy there, matey! The sky be blue 'cause of the way the sun's light scatters…"* — the prefix-end state reads out the prefix but does not causally carry it.

**Single-layer transfers do work** — 12 well-separated behavior families separate from the null at single layers (matched-query context-end L12–L17 and L20, across doses α = 1–4 and replace, F 0.19–0.33 vs nulls ≤ 0; plus one cross family at L15), and the independent resampling confirmation keeps single layers 13–17. But per pair they are hit-or-miss: pairs that transfer cleanly at the all-layer patch (F 0.69–0.77) can read 0.00 at L16/α = 1, while the L16 example above reads 0.94 — the family means (0.19–0.33) average strong flips with misses.

**Which matched-query pairs fail at context-end** (per-pair recount of the committed F tables at the all-layer full-state patch, A→B direction, one greedy draw per cell): the 15 pairs split cleanly by prefix pairing. All 5 bare↔conversation pairs are instrument failures, not transfer failures — anchor separation ≈ 0 (−0.01 to 0.22) because the conversation prefix's enthusiastic register never appears even in its own unpatched answers, so floor ≈ ceiling and F is degenerate (the source of #2094's early inflated 0.85–2.39 means). The gpu2 follow-up rebuilt them with a persistence-instructed replacement conversation prefix and recovered 4 of 5 (separations 0.51–1.02; all comparable parent-clean reads reproduce their direction there). Of the 10 well-separated pairs, 9 transfer at the all-layer patch (F 0.37–0.89, both bare→persona and persona→conversation); the one genuine laggard is bare→pirate on the story query (F = 0.14) — the patched story stays plain-register even though natively-prompted pirate stories express the register fine (separation 0.81), so creative-writing completions resist register injection harder than expository ones.

**Which information types are causally usable (#2162).** A patch-only sweep over 21 minimal-pair information types (facts, stated vs demonstrated policies, ICL task definitions, personas, discourse state; WildChat carriers) with a linear read probe alongside: the probe decodes almost everything from the natural states (55 of 76 cell × slot combos are stored-but-unusable), but the patch moves behavior for only **5 cells — all the stated-formatting-policy family at context-end** (`instr_format` plus its load and conflict variants, F 0.70–0.81 vs nulls ≈ 0.07–0.14). Retrievable facts (user's name, etc.), ICL task mappings, and demonstrated (as opposed to stated) policies are decodable but not causally usable through the single-position patch.

![Read × write 2×2: probe AUC vs causal F per type](https://raw.githubusercontent.com/superkaiba/explore-persona-space/20fcef9c282a97d6ae90473d54fc0ce5e59e26f5/figures/issue_2162/two_by_two.png)

## Result 3: our mapping does not predict the steering/patching effect — the response is not even linear

The transport cosine between the mapping's predicted shift f(v_C + Δ) − f(v_C) and the realized answer-vector shift is 0.00 at L20 (#1415; magnitude over-predicted ~16×) and tops out at 0.16 at context-end L14 in #2094 (pooled means ≤ 0.09) — prediction quality peaks exactly where patching works (context-end, mid-stack, matched query/cross) but never gets close to 1.

The failure is not specific to our fitted map: the edit → response relation is far from linear in general. Shift magnitude is flat in dose (log-log slope 0.00–0.06 vs 1.0 for a linear response), the shift direction drifts across doses (adjacent-dose cosine 0.62 at L14, despite split-half reliability 0.93 — real drift, not noise), and the best one-operator linear fit of edit → response, fitted on this experiment's own data, explains ≤ 8.4% of held-out variance. So the passively-fitted map fails to transport both because interventions leave its training distribution and because no linear operator describes the intervention response.

![Banked-map transport cosines, steered vs null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/result1b_transport_cosines.png)

![Per-pair map transport cosines at layer 20, centered on zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c78d145929017d6ff26285460cdb35322e523e71/figures/issue_1415/h2_map_transport_per_pair.png)

## Takeaways

1. The context vector is causally special but not causally sufficient: it is the only single position where activation edits move behavior past the null, yet even the maximal all-layer patch recovers only ~0.5–0.65 of a full context swap, and what transfers is the prefix's persona/register and stated formatting policy — not query identity, not stored facts.
2. Query identity lives in the query's own token states: replacing those (template-stripped) swaps which query is answered at up to F ≈ 0.95 while the prefix register survives — the two channels are separable.
3. Passive predictability ≠ causal control: the fitted linear map does not transport to interventions (transport cosine ≤ 0.16), and the intervention response is not linear at all (flat in dose, direction drifts, one-operator fit R² ≤ 0.084).

**Sources:** clean-result bodies of #1415 / #2094 / #2162; slot × layer heatmap dashboard `docs/issue2094_patching_dashboard.md`.
