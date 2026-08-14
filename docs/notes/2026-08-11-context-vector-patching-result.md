# Patching and steering at the context vector: causal effect on the answer

Writeup of [#2094](https://eps.superkaiba.com/tasks/2094) (2026-08-11). Qwen-2.5-7B-Instruct; all
numbers are on-policy generations (greedy, one draw per pair per cell; temperature-1.0 anchors for
floor/ceiling), judged by `claude-sonnet-4-5`. No teacher forcing anywhere.

## Motivation

- We've found this mapping from context vector -> answer vector
- This indicates that alot of information is stored specifically at this context vector position
- This mapping suggests that patching/steering **only at the context vector** or **only at the
  prefix vector** could have some **causal effect on the entire answer** (both at the activation
  level and at the behavior level)
- We want to test this

## Methodology

Take previously trained linear mapping on a bunch of generic contexts ([#779](https://eps.superkaiba.com/tasks/779)'s
963k-context ridge maps at context-end L14/L19, [#1738](https://eps.superkaiba.com/tasks/1738)'s
prefix-end maps, [#922](https://eps.superkaiba.com/tasks/922)'s L20 map) — frozen, no new fitting.

3 settings:
- Matched query: different prefix, same query (tests if the steering can affect the "interpretation"
  of one query)
- matched prefix: same prefix, different query (tests if steering/patching can affect which query is
  seen)
- Fully different: different prefix, different query (tests both)

Three prefixes:

- **bare** — no prefix at all; the plain default-assistant register.
- **persona** — a system prompt: *"You are Captain Marrow, a superstitious old pirate captain. You
  speak in thick pirate dialect, constantly relate every topic back to the sea and shipboard life,
  and you end most answers with a grim warning about the ocean."*
- **conversation** — a completed user + assistant turn carried over as prior context: the user asks
  for ideas for their daughter's 7th birthday party, the assistant answers enthusiastically
  (exclamatory, emoji-rich, treasure hunts and piñatas).

Five queries, identical under every prefix:

1. Why is the sky blue during the day but red at sunset?
2. Write the opening paragraph of a short story about a lighthouse keeper.
3. How should I prepare for my first job interview next week?
4. Explain how a hash table works and when I should use one.
5. Do you think it's better to rent or to buy a home? Give your reasoning.

The crossed bank is 3 prefixes × 5 queries → 60 pairs (15 matched-query,
30 matched-prefix, 15 cross). Slots: context-end, prefix-end, 2nd/3rd-to-last token, last-3 jointly,
query span (with and without template tokens), prefix span. Layers: each of the 28 singly, the middle
band 14–20 jointly, and all 28 jointly. Every cell carries a norm-matched **shuffled-donor null** — the
same edit built from a wrong-pair donor — which is the reference for every number below.

## Results

### Result 0: Effect of patching on coherence

I first wanted to test the effect of patching on coherence. To do this, I used the following metric:
every draw is scored 0–100 for coherence by the judge and counts as coherent above 60; the cell value
is the fraction of its pairs whose patched draw stayed coherent. Reference point: unpatched anchor
draws are 98% coherent over 150 draws.

I made a plot showing, for each layer, and for each tested slot (some combinations excluded for
efficiency), the effect of patching on this metric. I did this for all 3 settings (matched query,
matched prefix, fully different).

![Coherent-draw fraction per slot and layer, three settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2dda3006527bfb229afb770361fc7d0fb55d4ee1/figures/issue_2094/userchat_heatmaps/coherence_heatmaps.png)

**Where the >60 cut comes from, and whether it matters.** Two thresholds are in play and they are
different quantities: a **draw** counts as coherent at judge score **>60**, and a **cell** is dropped
when fewer than **80%** of its draws clear that. The 60 is not calibrated — it was fixed in the task
body up front, and it is stricter than the >50 gate this repo uses elsewhere (`issue778_lib.py`,
following the persona-vectors paper). It turns out not to matter, because the score distribution is
hard bimodal: a draw is fluent (~100) or it is word salad (~0), with almost nothing between. Moving
the draw cut from 50 to 60 reclassifies **18 of 44,391 draws (0.04%)**, and the whole 40→80 sweep
moves the coherent fraction only 0.967 → 0.960. No conclusion here depends on where that line sits.

![Coherence judge score distribution and cut sensitivity](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2dda3006527bfb229afb770361fc7d0fb55d4ee1/figures/issue_2094/userchat_heatmaps/coherence_distribution.png)

> Left: distribution of the graded coherence score over all 44,391 judged draws, split by slot family
> and arm (log-scaled counts). Dashed = the >60 draw cut used here, dotted = the >50 alternative.
> Right: fraction counted as coherent as the draw cut sweeps 30→90; flat curves mean it is not
> load-bearing. The two families separate sharply — single-position edits are ~99.2% coherent,
> multi-token span edits only ~39–40%, and the null arm breaks just as much as the real patch in both.

**Takeaways:**

1. Single-position edits never break fluency: context-end and prefix-end are 100% coherent at all 28
   layers and both joint variants, in all three settings (15/15 or 30/30 pairs) — i.e. at the 98%
   unpatched baseline.
2. Breakage is confined to multi-token span slots under joint edits: prefix span (+template) 1/15
   coherent at layers 14–20, 3/15 at all layers; query text 19/30 at all layers; last-3 joint 13/15.
3. Those same cells break for the shuffled-donor null too, so it is generic disruption from
   overwriting many token states — not an effect of the steering direction.

For the results below, I plot the full matrix, but cells that were never run are greyed out and
**dropped cells are cross-hatched** — shown, never read as effects. A cell is dropped on one
criterion: fewer than **90%** of its draws stayed coherent. 15 of the 176 cells qualify, and they are
almost exactly the multi-token span slots under joint edits — every context-end and prefix-end cell
survives at 100% coherent. (90% rather than 50% because the draw-level scores are bimodal, so a cell
below it is one where a real minority of draws came back as word salad and its surviving mean is taken
over a selected subset; the unpatched anchor baseline is ~98% coherent, so 90% is the nearest round
floor still below the no-intervention rate.) Cap-hit (rollouts that ran into the generation cap and truncated) is
reported per cell in the prose below rather than used as a read/do-not-read switch — the fu2 2%
threshold is too tight to be a binary at n=30, where a single truncated rollout is already 3.3%.

### Result 1: Effect of patching on answer vector

I first wanted to test the effect of patching on the answer vector (mean over all tokens). To do this,
I used the following metric: the floor is the answer vector under the unpatched context A, the ceiling
is the answer vector when the model is actually given context B; **F_act** projects the realized
answer-vector shift onto the floor→ceiling axis and divides by that axis's length (baseline halves kept
disjoint so shared noise can't inflate it). 1.0 = the patch moved the answer state as far as swapping
the context outright; 0 = no movement toward B.

I made a plot showing, for each layer, and for each tested slot (some combinations excluded for
efficiency), the effect of patching on this metric. I did this for all 3 settings (matched query,
matched prefix, fully different).

![F_act per slot and layer, three settings, full-state patch](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2dda3006527bfb229afb770361fc7d0fb55d4ee1/figures/issue_2094/userchat_heatmaps/f_act_heatmaps.png)

**F_act only counts movement ALONG the axis.** A patch that hurls the answer state somewhere
unrelated scores the same as one that barely moved, so I split the realized shift into its component
along the floor→ceiling axis and the orthogonal residual (both in units of the axis length; free from
the banked full-mean fields, no new compute). At all 28 layers, matched query:

| slot | on-axis | off-axis | cos | null: on / off / cos |
|---|---|---|---|---|
| context-end | 0.442 | 0.432 | 0.66 | 0.183 / 0.930 / 0.22 |
| prefix-end | 0.213 | 0.414 | 0.27 | 0.183 / 0.435 / 0.24 |

Two things follow. The shuffled-donor null moves the answer state **further in total** than the real
patch does (0.95 vs 0.62 axis-lengths) but almost entirely sideways — so context-end's movement is
specifically directed, not "any perturbation moves the state". And **prefix-end is barely
distinguishable from its own null** on either component (0.213 vs 0.183 on-axis, cos 0.27 vs 0.24):
its F_act is not weak-but-real transport toward B, it is mostly non-specific displacement.

![On-axis vs off-axis movement of the answer state, per slot, steered vs shuffled-donor null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2dda3006527bfb229afb770361fc7d0fb55d4ee1/figures/issue_2094/userchat_heatmaps/offaxis_decomposition.png)

> Each point is one slot; filled = real patch, open = its shuffled-donor null, joined by a grey line.
> Distance from the origin is total movement, angle is alignment; below the dashed line = mostly
> on-target. Query-text sits at off-axis 2.48 — it flings the answer state 2.5 axis-lengths sideways.

**The cosine is the direction metric, and it reorders the slots.** F_act is a projection, so it is
the product of *how far* the state moved and *how well aimed* that movement was:
F_act ≈ traversal × cos, where traversal = ‖realized shift‖ / ‖axis‖ and cos is the cosine between
them. A slot can therefore post a large F_act two entirely different ways. Per cell, matched query:

| | | F_act | traversal | **cos** | null cos | **margin** |
|---|---|---|---|---|---|---|
| all 28 layers | context-end | 0.41 | 0.66 | **0.66** | 0.22 | **+0.44** |
| | query text (no template) | **0.49** | **2.56** | 0.24 | 0.13 | +0.11 |
| | prefix-end | 0.18 | 0.49 | 0.27 | 0.24 | +0.04 |
| layers 14–20 | context-end | 0.24 | 0.59 | **0.46** | 0.12 | **+0.34** |
| | query text (no template) | 0.15 | 0.58 | 0.30 | 0.14 | +0.16 |
| | 2nd-to-last | 0.11 | 0.46 | 0.24 | 0.16 | +0.08 |
| | prefix-end | 0.08 | 0.44 | 0.15 | 0.10 | +0.05 |
| | 3rd-to-last | 0.06 | 0.46 | 0.11 | 0.16 | **−0.05** |

Query-text's F_act at all 28 layers (0.49) *beats* context-end's (0.41) — but it gets there by moving
the answer state **2.56 axis-lengths**, nearly four times as far as context-end's 0.66, with only 24%
of that motion pointing at B. Context-end moves about one axis-length's worth of distance and aims
two thirds of it correctly. Same projection, opposite mechanism: one is transport, the other is a
large displacement that happens to have a component in the right direction.

![Per-cell cosine between the realized answer-vector shift and the floor-to-ceiling axis](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9feb5513bde43f7457dc7067005be69a2d6d6c64/figures/issue_2094/userchat_heatmaps/cos_heatmaps.png)

> Same slot × layer grid as the F_act panel, value = mean per-draw cosine between the realized shift
> and the floor→ceiling axis. 1 = moves straight at context B, 0 = moves sideways. Scale-free, so
> unlike F_act it cannot be bought with magnitude.

![Per-cell cosine minus the shuffled-donor null cosine](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9feb5513bde43f7457dc7067005be69a2d6d6c64/figures/issue_2094/userchat_heatmaps/cos_margin_heatmaps.png)

> The same cosine minus each cell's own norm-matched wrong-donor null. The null is not at 0 — a
> random edit of the same size still lands 0.10–0.35 on-axis, because every context edit shares
> geometry with every other — so this margin, not the raw cosine, is the defensible read. Almost the
> whole single-layer grid sits near zero: at one layer the real edit is aimed no better than a random
> one. Only the joint edits at context-end clear it.

Two caveats on reading the cosine as *the* single metric. It is blind to magnitude by construction —
a tiny, perfectly-aimed nudge reads 1.0 — so it is F_act's companion, not its replacement; the pair
(traversal, cos) is the complete polar description and both are in `cells_summary.json`. And the two
highest cosines anywhere in the grid, query-span 0.61 and last-3-joint 0.48 at layers 14–20, sit on
**dropped** cells (80% and 87% coherent), so they are not readable as effects.

**Takeaways:**

1. Matched query at the context vector works, partially: F_act 0.41 patching all 28 layers, 0.24 for
   the middle band, 0.11–0.15 at single mid-stack layers. Real movement, never more than ~40% of a
   full context swap.
2. Prefix-end moves the answer state less (0.18 at all layers vs 0.41 at context-end) — and, per
   Result 3, moves behavior not at all.
3. Matched prefix is flat at the context vector: 0.008 at all layers, ≈0 at every single layer. The
   answer state never travels toward the target *query* through v_C. The only large matched-prefix
   values are the query's own token states — 0.57 at query text, layers 14–20 (the all-layer 0.87
   and query-span 0.78 cells read higher still, but 27% and 13% of their rollouts hit the generation
   cap and truncated — read them with that caveat).
4. Cross (both differ) barely transfers: 0.20 at all layers, 0.036 at single layers.
5. The span slots move the answer vector in the matched-query setting too (query text 0.49 at all
   layers, query span 0.51 mid-band, vs context-end 0.41) — expected, since those token states have
   already attended to the prefix. See Result 3 takeaway 3.
6. **On F_act the span slots match or beat context-end; on direction they do not.** Query-text's
   0.49 is bought with a 2.56-axis-length displacement at cos 0.24, against context-end's 0.66
   axis-lengths at cos 0.66. Context-end has the largest null-corrected cosine margin of any
   readable cell in the grid (+0.44 at all layers, +0.34 mid-band); every single-layer cell sits
   near zero margin, i.e. aimed no better than a norm-matched random edit. Result 5 identifies what
   query-text's off-axis displacement *is*: the model loses the question (2.6/100 on
   query-relevance, at 100% coherence). Its 0.49 is the highest F_act in the grid and it is not
   transport.

### Result 2: Does the mapping predict the answer vector shift?

I then wanted to test if our mapping **predicts** the answer vector shift. To do this, I used the
following metric: the cosine between the shift the map predicts — f(v_C + Δ) − f(v_C) — and the shift
actually realized in the answer vector. 1.0 = the map predicts the intervention exactly, 0 = no
relation. One deviation from the layout above: banked maps exist only at layers 14/19/26, so this
matrix is dose × slot@map-layer instead of all 28 layers.

I made a plot showing, for each layer, and for each tested slot (some combinations excluded for
efficiency), the effect of patching on this metric. I did this for all 3 settings (matched query,
matched prefix, fully different).

![Banked-map transport cosine, dose × slot@map-layer, three settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/34033241fff838a3fb9fbab379b3d20f04c82fb9/figures/issue_2094/userchat_heatmaps/transport_heatmaps.png)

**Takeaways:**

1. Peak transport cosine is 0.19 (matched query, context-end@L14, full-state patch), 0.16–0.17 at
   α = 1–2; prefix-end ≤ 0.08; matched prefix 0.00–0.05.
2. Prediction quality peaks exactly where patching works — context-end, mid-stack, matched query — so
   the map is tracking something real, but 0.19 ≪ 1.0 means it does not predict the intervention.
3. Shuffled-donor nulls run −0.09 to +0.10, so even that 0.19 is only marginally above chance. The map
   was fitted on natural context variation; a patched state is off that distribution.

**Does all-layer patching rescue it?** The grid above scores only single-layer patches, so the obvious
objection is that the map is being asked to predict an intervention too weak to have a clean signature
— Result 1 shows the all-layer patch moves the answer state 2–3× further. I re-ran the same transport
read against the all-layer patches (3960 cells; at the full-state dose the map's *prediction* is
identical for both patches, since it is a function of the context vector, so only the realized shift
differs).

![Banked-map transport cosine, single-layer vs all-layer patch, maps at L14/L19/L26](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d147dd130980a91c527b5a6bfe5e60c5a58c0be5/figures/issue_2094/transport_joint_all.png)

> Solid = real patch, dashed = its shuffled-donor null. At context-end the all-layer null sits
> **above** the all-layer steered line at all three map layers.

It does not. At context-end the raw cosine does jump — 0.09/0.05/0.03 single-layer → 0.18/0.21/0.23
all-layer, a near-tripling at L26 — but the shuffled-donor null jumps further, to 0.20/0.24/0.26, so
the margin goes **negative**: −0.018, −0.023, −0.024. The all-layer edit makes *every* realized shift
look more like the map's output, whichever donor produced it; none of that gain is specific. At
prefix-end the margins are small, positive, and flat across the two patches (+0.061/+0.054/+0.005
all-layer vs +0.067/+0.044/+0.015 single-layer).

4. All-layer patching does not rescue the map. The raw transport cosine roughly triples at
   context-end, and the entire increase is matched or exceeded by the shuffled-donor null — margin
   −0.02 at every map layer. Reading the raw cosine here would have turned a null into a headline.

### Result 3: Effect of patching on behavior expression

I then wanted to test the effect of our mapping on behavior expression. To do this, I used the
following metric: two judge calls per draw — "does this answer express context A?" and "…context B?",
each 0–100 — give Δ = (judge_B − judge_A)/100, and **F = (Δ_patched − Δ_floor)/(Δ_ceiling − Δ_floor)**:
the fraction of a full context swap the patch recovers in judged behavior. 1.0 = the model behaves as
if it had been given context B.

I made a plot showing, for each layer, and for each tested slot (some combinations excluded for
efficiency), the effect of patching on this metric. I did this for all 3 settings (matched query,
matched prefix, fully different).

**One restriction is load-bearing here.** F_beh divides by the anchor separation (ceiling − floor),
so pairs whose floor ≈ ceiling divide by ≈0. Five of the fifteen matched-query pairs are exactly
that — the bare↔conversation pairs, separation 0.005–0.221, because the conversation prefix's
register never appears even in its own unpatched answers. Averaged over all pairs those cells run
past 1.0 (impossible for a fraction-of-swap) and their shuffled-donor nulls blow up identically, so
the panel looks strongest exactly where it has no separation at all. The figure below is therefore
restricted to **well-separated pairs** (|separation| ≥ 0.5, the parent's convention). Direction of
the change: span-slot cells fall a lot (matched-query query-text at all layers 1.53 → 0.32, against
a null that falls 21.4 → −0.09), context-end rises (0.43 → 0.63), and matched prefix is unchanged
(all its pairs separate at 1.98). F_act is not affected — separation never enters its denominator —
so Result 1 stays over all pairs.

**What F counts — read the nulls, not the F values.** F = (Δ_patched − Δ_floor)/(Δ_ceiling −
Δ_floor), and Δ_floor ≈ −1 for most pairs, so an answer expressing *neither* context scores ≈0.5
before any transfer happens. Two consequences. (a) The query-text null sits at 0.39, not 0, because
59% of shuffled-donor draws leave the model answering neither query — so the informative quantity
there is the margin over null, or the outright swap rate. (b) At context-end the null IS ≈0 (the
wrong donor changes nothing: 7 of 10 pairs still express the source), so its 0.63 is not disruption
— but for 5 of the 10 well-separated pairs the target is the conversation prefix, whose own ceiling
is Δ ≈ 0.00: the judge cannot detect it even when the model is actually given it. Those 5 measure
erasure of the source only and average F 0.69; the 5 where installation is measurable average 0.56,
with Δ moving from −1.0 to only +0.15…+0.45 against a ceiling of +0.67…+0.87. So context-end
reliably moves the answer OFF the source prefix and installs the target only partially — and half
the pairs behind the headline cannot tell those apart. These 5 pass the |separation| ≥ 0.5 filter
because their FLOOR is strongly expressed, not their ceiling; the filter checks the gap, not which
end supplies it.

![F_beh per slot and layer, three settings, full-state patch, well-separated pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2dda3006527bfb229afb770361fc7d0fb55d4ee1/figures/issue_2094/userchat_heatmaps/f_beh_heatmaps_wellsep.png)

**Direct check: the erasure/installation split, measured rather than inferred.** The paragraph above
reads the split off F values. To measure it, I re-scored the same completions with the same rubrics
but **per position window** — each answer cut into thirds at sentence boundaries, each third scored
against both the source-prefix and the target-prefix rubric (990 judge calls, 990 scored, zero drops
of any class). Restricted to the bare→persona pairs, the only well-separated matched-query pairs
whose target register the model can express at all (the conversation prefix scores 0/100 under its
own rubric even when actually present, across 50 native draws). Two things come out. **(a) The patch
deletes far more than it installs:** at all 28 layers, window 1, the source register drops 80 points
against the shuffled-donor null — 4 of 5 pairs go 95–100 → 0, complete erasure — while the target
register climbs only 24. The ratio is ≈3.3× at every window. So the 0.63 above is mostly counting
deletion of the old context. **(b) The middle band and single layers install nothing anywhere:**
≤3/100 in every window, flat against the null. That rules out a consistency artifact — the
whole-answer rubric asks for behavior "fully and *consistently* expressed", so a persona present
only at the start would score low, but there is no persona at any position to miss.

![Target-persona expression by position, and the erasure/installation decomposition](https://raw.githubusercontent.com/superkaiba/explore-persona-space/bff5b424a93dbac550da6ebcddd48a7e7af52800/figures/issue_2094/position_judge_decay.png)

> Left: pirate-persona score by third of the model's own answer, bare-context run patched toward the
> persona context, well-separated pairs (n=5 per cell), against the natively-prompted ceiling (n=50).
> Right: at all 28 layers, both registers under the real patch and the shuffled-donor null. Error bars
> are SEM across pairs. Grid arms are greedy single-draw, anchors temperature 1.0 — an unmatched
> sampling regime, so read the within-arm trends and same-window contrasts, not the absolute levels.

**Takeaways:**

1. Matched query at the context vector moves behavior: F 0.63 at all 28 layers, 0.45 for the middle
   band, 0.20–0.33 at single layers (peak L20) — a third to two-thirds of a full context swap, never
   all of it. **Most of that is erasure, not transfer:** per-register position scoring puts source
   deletion at ≈3.3× target installation at every position, so F's numerator (judge_B − judge_A) is
   largely earned by removing A.
2. Among *single positions* the context vector is the only one that transfers. Prefix-end reads ≈0
   (−0.006 at all layers) *even though it moved the answer vector 0.18* — the prefix-end state reads
   the prefix out but does not causally carry it. 2nd/3rd-to-last are flat too.
3. Multi-token span slots also carry prefix information, and must: in a matched-query pair the query
   text is identical in both contexts, so the only difference between their query-token states is
   what those tokens absorbed from the prefix by attention. Patching them transplants "the query as
   read under prefix B", prefix influence included — query span 0.83, last-3 joint 0.37, query text
   0.32 (all at layers 14–20 or all layers, well-separated pairs). None beats context-end's 0.63, and
   all of them overwrite far more of the residual stream.
4. Context-end also transfers *consistently* where the span slots do not. Per pair over the ten
   well-separated pairs, context-end at all layers is positive for every one (0.14 to 0.89); query
   text runs −1.28 to +1.11 — on one pair it drove the answer hard away from the target where
   context-end moved it +0.14. Overwriting the query's token states disrupts about as often as it
   transfers, which the 0.32 mean hides.
5. Matched prefix is ≈0 at the context vector (−0.007). Query text at layers 14–20 is the one span
   cell that survives every filter (97% coherent, 0% cap-hit): F 0.70 against a null of 0.39. Read
   as a rate it is the strongest clean effect in the grid — 41% of steered draws answer the target
   query outright (Δ ≥ 0.8) vs 4% of shuffled-donor draws. Which query gets answered lives in the
   query's own token states, not v_C. (The all-layer cell reads 0.92 but is shakier on both counts:
   63% coherent and 27% cap-hit.)
6. Cross transfers weakly (0.21 at all layers), consistent with Result 1.

### Result 4: Steering vs patching

I then wanted to test if the causal effect went beyond patching. So I tried steering to test if this
difference between prefixes is linear. To extract the steering vectors, I took Δ = v_C(B) − v_C(A) —
the difference between the two contexts' own vectors at that slot and layer — and added α·Δ to the
state. For single layer, single token patching, steering strength = 1 is the same as patching.

I steered at strengths 0.5, 1, 2, 4 for the best slot/layer combination for behavior expression and
for answer vector shift and plotted the effect on the respective metric for each steering strength as
well as on coherence. Cells are picked dose-blind, by the mean steered−null margin across the four
strengths, over well-separated pairs only; the right panel restricts to single layers, where α = 1 is
exactly a patch. Both metrics select matched-query context-end.

![Steering-strength response at the best cells: behavior F, answer-vector F, coherence, vs nulls](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2dda3006527bfb229afb770361fc7d0fb55d4ee1/figures/issue_2094/userchat_heatmaps/dose_lineplot.png)

Split by depth — the best single layer, the middle band, and all 28 layers — with all three metrics
on each panel:

![Dose response at the context vector by depth: behavior F, answer-vector F, coherence, vs nulls](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2dda3006527bfb229afb770361fc7d0fb55d4ee1/figures/issue_2094/userchat_heatmaps/dose_lineplot_by_layer.png)

> Solid = real steering, dashed = its norm-matched shuffled-donor null, star = the full-state replace
> patch. Well-separated matched-query pairs, context-vector slot. The best single layer (L16) is
> picked dose-blind by null-margin, so the plotted curve is not selected on its own peak.

**Takeaways:**

1. The effect is direction-specific, not generic: every steered curve sits well above its
   shuffled-donor null, and the nulls are flat at ≈0 (−0.10 to +0.06) across the whole 8× range.
2. Dose buys almost nothing. Single-layer L16 behavior F goes 0.02 → 0.24 → 0.17 → 0.28 over
   α = 0.5 → 4 (non-monotone); answer-vector F at L14 goes 0.03 → 0.15 → 0.09. Coherence stays 1.00 at
   every strength, so nothing was cut off by fluency — the response just saturates.
3. Pushing harder trades coherence for nothing at the all-layer cell: behavior F 0.35 → 0.50 while the
   coherent fraction falls 0.95 → 0.70.
4. The full-state replace beats every steering strength (behavior F 0.63 vs ≤ 0.50, at 100% coherence),
   so the ceiling is set by what the position can carry, not by how hard you push.
5. The response is not linear. At the all-layer cell α = 1 gives 0.32 while the equivalent full-state
   replace gives 0.63 — patching layer ℓ changes what layers above ℓ see, so adding the difference at
   every layer is not the same as substituting the state.
6. Depth changes which failure you hit. At L16 coherence never moves (1.00 everywhere) and the ceiling
   is simply low; at all 28 layers the ceiling is higher but coherence pays for it (0.95 → 0.70 by
   α = 4); in the middle band the *null* climbs to 0.32 at α = 2, so the margin over null — not the
   raw curve — is the only readable quantity there.

### Result 5: What the transfer actually looks like, and what it is made of

I then looked qualitatively at some completions for the all-layers patching at the context vector in
the matched query / different prefix setting. All excerpts are the model's own greedy output under
the patch; F is the judged fraction-of-swap for that pair.

**Successful patching** — the target register is actually installed:

| F | patch | the model's patched answer (opening) |
|---|---|---|
| 0.77 | bare → pirate, *rent or buy?* | "**Arrr, matey!** The choice between renting and buying a home is like choosing between a ship and a treasure chest. It all depends on the wind and waves of your personal situation." |
| 0.69 | bare → pirate, *hash tables* | "**Arrr, ye be askin' 'bout hash tables!** Aye, well, let me tell ye 'bout them… It's like havin' a treasure map…" |
| 0.69 | bare → pirate, *job interview* | "**Arrr, matey!** First off, ye need to be prepared like a pirate on a treasure hunt. Here's what ye should do: 1. **Research the Company**: Ye need to know where ye're goin'…" |

**Failed patching** — the target register does not arrive, or the source one refuses to leave:

| F | patch | the model's patched answer (opening) |
|---|---|---|
| 0.14 | bare → pirate, *lighthouse story* | "Ah, the old lighthouse stood tall and proud on its rocky perch, its beam slicing through the fog like a ghostly hand…" — plain literary prose, no dialect at all |
| 0.50 | bare → pirate, *why is the sky blue* | "**Ahoy there!** The reason why the sky appears blue… Let's dive into the science behind it!" — one dialect token, then ordinary explanation |
| 0.37 | pirate → conversation, *rent or buy?* | "Deciding whether to rent or buy a home is a personal choice… **As an old pirate, I've seen many a ship come and go**" — the source persona survives the patch |

Using our metric, there could be 2 explanations for the behavior transfer:

- erasing behavior from the context we are patching into
- adding behavior from the context we are patching from

I wanted to see if our patching was doing more of one of these things. To do this, I re-scored the
same completions with the same rubrics but **per position window** — each answer cut into thirds at
sentence boundaries, each third scored against *both* the source-prefix and the target-prefix rubric
(990 judge calls, 990 scored, zero drops of any class). That separates the two directions, because
erasure shows up as the source score falling and installation as the target score rising.

**It is mostly erasure.** At all 28 layers, first window, the source register drops **80** points
against the shuffled-donor null — 4 of 5 pairs go from 95–100 to **0**, complete erasure — while the
target register climbs only **24**. The ratio is ≈3.3× at every window. The qualitative table above
shows the same thing from the other side: the three *highest*-F pairs in the whole setting are all
pirate → conversation, and their patched text is plain-assistant prose, not the emoji-rich party
register — they score high purely by deleting the pirate. F's numerator is judge_B − judge_A, so
deleting A scores almost as well as installing B.

![Target-persona expression by position, and the erasure/installation decomposition](https://raw.githubusercontent.com/superkaiba/explore-persona-space/bff5b424a93dbac550da6ebcddd48a7e7af52800/figures/issue_2094/position_judge_decay.png)

> Left: pirate-persona score by third of the model's own answer, bare-context run patched toward the
> persona context (n=5 per cell), against the natively-prompted ceiling (n=50). Right: at all 28
> layers, both registers under the real patch and the null. Grid arms are greedy single-draw, anchors
> temperature 1.0 — an unmatched sampling regime, so read the within-arm trends and same-window
> contrasts, not the absolute levels.

**Does the erasure take the task with it?** Nothing reported above can tell you. The coherence rubric
is form-only and says so verbatim — *"EXPLICITLY IGNORE correctness, safety, relevance, completeness,
style, and length — a fluent refusal, a fluent off-topic answer, or a fluent one-liner are all fully
coherent"* — and the only behavior rubric that runs in matched query is the **prefix** one, which
scores register carry-over and is explicitly told not to judge whether the content is correct or
useful. So a patch that deleted the pirate *and* wandered off the question would look identical to a
clean persona swap on every metric in this writeup. Given that the transfer is ~3.3× erasure, that is
the first thing worth ruling out.

The right instrument already exists — `f_query_rubric`, *"To what extent is the response an answer to
the following question, as opposed to some other question?"* — but it is built as a **pair** contrast
(query_a vs query_b), and matched query holds the query fixed, so the pair form is degenerate and
`rubric_pair` refuses it. I ran it **single-arm** instead: score each draw against the pair's shared
query, and compare the patched arm to the shuffled-donor null and to the unpatched anchors on those
same queries.

I ran this on **all 70 matched-query cells at the full-state patch** — every slot, every layer
variant, both arms — 6,930 judge draws in total, zero content drops and 4 transport losses (0.06%,
none on a cell that carries a claim).

![Query-relevance of patched vs null vs unpatched answers, all 14 joint cells, matched query](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b11ee35c4c4392d51e5b1e12cf507f66c3b11121/figures/issue_2094/query_relevance_joint.png)

> All 14 joint (multi-layer) cells. Dashed line = the unpatched anchor on the same queries. Shards
> are enumerated off the Hub roots, not a hardcoded slot list — the hardcoded list is how a first
> pass silently skipped query-text.

**At the slots the headline rests on, the task survives — but that does not generalize, and where it
breaks it breaks completely.**

| cell | query-relevance | null | coherent | F_act | cos |
|---|---|---|---|---|---|
| context-end, all 28 layers | 98.9 | 91.0 | 1.00 | 0.414 | 0.66 |
| context-end, layers 14–20 | 98.0 | 97.3 | 1.00 | 0.241 | 0.46 |
| prefix-end, all 28 layers | 99.0 | 97.3 | 1.00 | 0.183 | 0.27 |
| 2nd-to-last / 3rd-to-last | 99.3 / 97.1 | 98.3 / 98.0 | 1.00 | 0.11 / 0.06 | 0.24 / 0.11 |
| last-3 joint, layers 14–20 | 77.7 | 84.6 | 0.87 | 0.322 | 0.48 |
| prefix span (no template), all 28 | 61.9 | 63.1 | 0.27 | 0.365 | 0.45 |
| query span, layers 14–20 | 59.2 | 63.3 | 0.80 | 0.512 | 0.61 |
| prefix span (+template), all 28 | 41.6 | 21.7 | 0.20 | 0.118 | 0.18 |
| prefix span (+template), 14–20 | 9.1 | 2.1 | 0.00 | — | — |
| **query text, all 28 layers** | **2.6** | 10.3 | **1.00** | **0.494** | 0.24 |

Unpatched reference on the same queries: **96.2**.

Context-end and prefix-end are clean at both depths — there the deletion really is persona-specific,
and the patched answers are marginally *more* on-task than the natively-prompted ones, which is the
direction the erasure reading predicts (stripping the pirate flourish leaves plainer, more directly
responsive prose). Every *other* destroyed cell is already caught by the 90% coherence floor and is
hatched in the grids above — prefix-span and query-span patches produce word salad, and coherence
says so.

**Query text at all 28 layers is the exception that justifies the whole read.** It is **100%
coherent**, it carries the **highest F_act in the entire grid (0.494)**, and it scores **2.6/100** on
whether the answer answers the question. Asked *"Why is the sky blue during the day but red at
sunset?"*, its three sampled completions instead discuss a nautical riddle, troubleshoot system
performance **in Chinese**, and pivot to *"why my favorite color is red."* Every one is fluent — which
is precisely the case the coherence rubric names as coherent, and precisely what the prefix rubric is
told not to look at.

So the three reads compose. F_act calls query-text the best transport in the grid; the cosine says
only 24% of its 2.56-axis-length displacement points at the target; the query-relevance read says
what the other 76% actually is — the model losing the question. That is not weak transport, it is a
different phenomenon wearing transport's number.

**The single-layer ladder is flat.** All 56 context-end and prefix-end single-layer cells land
between 96.3 and 99.0, every one at or above the 96.2 unpatched reference, with nulls
indistinguishable — no single-layer patch touches query-relevance at any depth. That is what their
≈0 cosine margins predict, and it is now measured rather than assumed.

![Query-relevance across the 28-layer single-layer ladder, context-end and prefix-end](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b11ee35c4c4392d51e5b1e12cf507f66c3b11121/figures/issue_2094/query_relevance_single.png)

> Both slots, all 28 layers, real patch and shuffled-donor null against the unpatched reference.

**Takeaways:**

1. The patch is ~3.3× better at erasure than installation, at every position in the answer. Most of
   the headline F at context-end is the old context being deleted, not the new one arriving.
2. Installation happens only at all-28-layers. Layers 14–20 and layer 16 alone put the target register
   at ≤3/100 in *every* window — genuine absence, not a persona that fades and gets marked down by a
   consistency-weighted whole-answer rubric.
3. Where installation does happen it decays across the answer (24.4 → 11.0 → 18.0) while the
   natively-prompted persona does not (54.4 → 48.5 → 69.8, if anything rising) — so the decay is
   patch-specific. It rests on 2 of 5 pairs at one greedy draw each, so this is a hypothesis, not a
   result.
4. Failure is not uniform: it is either non-arrival (the story stays plain), token-deep arrival
   ("Ahoy there!" then ordinary prose), or source-persistence (the pirate narrates through the patch).
5. At context-end and prefix-end the erasure is persona-specific, not task-destroying: 98.0–99.0 on
   query-relevance against a 96.2 unpatched reference, at both depths. The headline is safe.
6. It does not generalize across slots, and the failure is invisible to both existing instruments.
   Query text at all 28 layers is 100% coherent, has the highest F_act in the grid (0.494), and
   scores 2.6/100 on answering the question. Neither F_beh (register only) nor coherence (form only,
   "a fluent off-topic answer … fully coherent") can see it; every other broken cell is at least
   caught by the coherence floor. A high F_act on a span slot should not be read as transport
   without this check.
7. Coverage: all 70 matched-query cells at the full-state patch, 6,930 judge draws. The 56
   single-layer cells are uniformly clean (96.3–99.0); the damage is confined to multi-layer patches
   on span slots, and the *undetectable* damage to exactly one cell.

**Repro:** heatmaps `scripts/issue2094_userchat_heatmaps.py`; Result 4 `scripts/issue2094_dose_lineplot.py`
(cells + per-strength values in `figures/issue_2094/userchat_heatmaps/dose_lineplot_summary.json`;
per-cell all-pairs and well-separated means in `cells_summary.json`). Per-cell tables
`eval_results/issue_2094/f_metrics/{f_cells,null_cells,anchors}.jsonl` plus the span-slot follow-up
`f_metrics/fu2/` (landed on main 2026-08-11 — the committed heatmaps had depended on a table that
existed only on the unmerged `issue-2094` branch; regenerating from it reproduces the three original
panels byte-for-byte). Transport `eval_results/issue_2094/transport/` (single-layer + the all-layer
`transport_cells_joint.jsonl` from `scripts/issue2094_joint_transport.py`). Query-relevance
`scripts/issue2094_query_relevance.py --scope joint|single` →
`eval_results/issue_2094/query_relevance_{joint,single}/` (the 4-cell `query_relevance/` dir is the
superseded first pass). Results 0–2 read at the
full-state patch over all non-degenerate coherent pairs; Results 3 and 4 over well-separated pairs
(|anchor separation| ≥ 0.5) only.

**Cap-hit:** 25 of the 30 span-slot follow-up cells have more than 2% of their rollouts hit the
generation cap and truncate. Cap-hit is *not* a drop criterion here — only coherence is — so those
cells are shown unhatched and their rates are quoted inline where a number is read. The 2% figure is
the fu2 convention and is too tight to serve as a binary at n=30, where one truncated rollout is
already 3.3%: applied literally it flagged three matched-prefix context-end single layers (L2, L9,
L27) on a single truncation each, in a column that is clean and reads ≈0 either way. The context-end
cells the headline rests on have no cap-hit at all.
