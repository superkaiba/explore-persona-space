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

I tried a variety of prefixes and queries ([see dashboard here](https://github.com/superkaiba/explore-persona-space/blob/main/docs/issue2094_patching_dashboard.md))
including persona prompts, ICL examples, and random wildchat prefixes (that indicate something at
least special about the user). The crossed bank is 3 prefixes × 5 queries → 60 pairs (15 matched-query,
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

![Coherent-draw fraction per slot and layer, three settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/985a61c307b19472f6266693bb4f4753d7645789/figures/issue_2094/userchat_heatmaps/coherence_heatmaps.png)

**Takeaways:**

1. Single-position edits never break fluency: context-end and prefix-end are 100% coherent at all 28
   layers and both joint variants, in all three settings (15/15 or 30/30 pairs) — i.e. at the 98%
   unpatched baseline.
2. Breakage is confined to multi-token span slots under joint edits: prefix span (+template) 1/15
   coherent at layers 14–20, 3/15 at all layers; query text 19/30 at all layers; last-3 joint 13/15.
3. Those same cells break for the shuffled-donor null too, so it is generic disruption from
   overwriting many token states — not an effect of the steering direction.

For the results below, I plot the full matrix, but cells that were never run are greyed out and
**dropped cells are cross-hatched** — shown, never read as effects. A cell is dropped on any of three
criteria: fewer than half its draws stayed coherent, more than 2% of its rollouts hit the generation
cap and truncated (the fu2 compromised-family threshold), or fewer than 5 pairs survive. 19 of the
176 cells qualify, nearly all of them span slots under joint edits.

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

![F_act per slot and layer, three settings, full-state patch](https://raw.githubusercontent.com/superkaiba/explore-persona-space/985a61c307b19472f6266693bb4f4753d7645789/figures/issue_2094/userchat_heatmaps/f_act_heatmaps.png)

**Takeaways:**

1. Matched query at the context vector works, partially: F_act 0.41 patching all 28 layers, 0.24 for
   the middle band, 0.11–0.15 at single mid-stack layers. Real movement, never more than ~40% of a
   full context swap.
2. Prefix-end moves the answer state less (0.18 at all layers vs 0.41 at context-end) — and, per
   Result 3, moves behavior not at all.
3. Matched prefix is flat at the context vector: 0.008 at all layers, ≈0 at every single layer. The
   answer state never travels toward the target *query* through v_C. The only large matched-prefix
   values are the query's own token states — 0.57 at query text, layers 14–20 (the all-layer 0.87
   and query-span 0.78 cells are dropped: 27% and 13% cap-hit).
4. Cross (both differ) barely transfers: 0.20 at all layers, 0.036 at single layers.
5. The span slots move the answer vector in the matched-query setting too (query text 0.49 at all
   layers, query span 0.51 mid-band, vs context-end 0.41) — expected, since those token states have
   already attended to the prefix. See Result 3 takeaway 3.

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

![F_beh per slot and layer, three settings, full-state patch, well-separated pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/985a61c307b19472f6266693bb4f4753d7645789/figures/issue_2094/userchat_heatmaps/f_beh_heatmaps_wellsep.png)

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
   query's own token states, not v_C. (The 0.92 all-layer cell is dropped: 63% coherent, 27%
   cap-hit.)
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

![Steering-strength response at the best cells: behavior F, answer-vector F, coherence, vs nulls](https://raw.githubusercontent.com/superkaiba/explore-persona-space/94728abb5029fe49d9d632633f5a5e6b451b65d3/figures/issue_2094/userchat_heatmaps/dose_lineplot.png)

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

**Repro:** heatmaps `scripts/issue2094_userchat_heatmaps.py`; Result 4 `scripts/issue2094_dose_lineplot.py`
(cells + per-strength values in `figures/issue_2094/userchat_heatmaps/dose_lineplot_summary.json`;
per-cell all-pairs and well-separated means in `cells_summary.json`). Per-cell tables
`eval_results/issue_2094/f_metrics/{f_cells,null_cells,anchors}.jsonl` plus the span-slot follow-up
`f_metrics/fu2/` (landed on main 2026-08-11 — the committed heatmaps had depended on a table that
existed only on the unmerged `issue-2094` branch; regenerating from it reproduces the three original
panels byte-for-byte). Transport `eval_results/issue_2094/transport/`. Results 0–2 read at the
full-state patch over all non-degenerate coherent pairs; Results 3 and 4 over well-separated pairs
(|anchor separation| ≥ 0.5) only.

**Cap-hit:** 25 of the 30 span-slot follow-up cells are cap-hit compromised (>2% of their rollouts
hit the generation cap and truncate). Those cells are cross-hatched in the panels along with the
incoherent and thin-n ones. Three matched-prefix context-end single layers (L2, L9, L27) are hatched
on a marginal 1-of-30 truncation — the 2% rule applied literally; the rest of that column is clean
and reads ≈0 either way. The context-end cells the headline rests on are not cap-hit affected.
