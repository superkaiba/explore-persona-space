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

![Coherent-draw fraction per slot and layer, three settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/94728abb5029fe49d9d632633f5a5e6b451b65d3/figures/issue_2094/userchat_heatmaps/coherence_heatmaps.png)

**Takeaways:**

1. Single-position edits never break fluency: context-end and prefix-end are 100% coherent at all 28
   layers and both joint variants, in all three settings (15/15 or 30/30 pairs) — i.e. at the 98%
   unpatched baseline.
2. Breakage is confined to multi-token span slots under joint edits: prefix span (+template) 1/15
   coherent at layers 14–20, 3/15 at all layers; query text 19/30 at all layers; last-3 joint 13/15.
3. Those same cells break for the shuffled-donor null too, so it is generic disruption from
   overwriting many token states — not an effect of the steering direction.

For the results below, I plot the full matrix, but the cells that were never run are greyed out, and
cells where fewer than half the draws stayed coherent are marked `*` — they are shown, never read as
effects.

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

![F_act per slot and layer, three settings, full-state patch](https://raw.githubusercontent.com/superkaiba/explore-persona-space/94728abb5029fe49d9d632633f5a5e6b451b65d3/figures/issue_2094/userchat_heatmaps/f_act_heatmaps.png)

**Takeaways:**

1. Matched query at the context vector works, partially: F_act 0.41 patching all 28 layers, 0.24 for
   the middle band, 0.11–0.15 at single mid-stack layers. Real movement, never more than ~40% of a
   full context swap.
2. Prefix-end moves the answer state less (0.18 at all layers vs 0.41 at context-end) — and, per
   Result 3, moves behavior not at all.
3. Matched prefix is flat at the context vector: 0.008 at all layers, ≈0 at every single layer. The
   answer state never travels toward the target *query* through v_C. The only large matched-prefix
   values are the query's own token states (query text 0.87, query span 0.78) — and those sit on
   coherence-compromised draws (19/30, 27/30).
4. Cross (both differ) barely transfers: 0.20 at all layers, 0.036 at single layers.

### Result 2: Does the mapping predict the answer vector shift?

I then wanted to test if our mapping **predicts** the answer vector shift. To do this, I used the
following metric: the cosine between the shift the map predicts — f(v_C + Δ) − f(v_C) — and the shift
actually realized in the answer vector. 1.0 = the map predicts the intervention exactly, 0 = no
relation. One deviation from the layout above: banked maps exist only at layers 14/19/26, so this
matrix is dose × slot@map-layer instead of all 28 layers.

I made a plot showing, for each layer, and for each tested slot (some combinations excluded for
efficiency), the effect of patching on this metric. I did this for all 3 settings (matched query,
matched prefix, fully different).

![Banked-map transport cosine, dose × slot@map-layer, three settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/94728abb5029fe49d9d632633f5a5e6b451b65d3/figures/issue_2094/userchat_heatmaps/transport_heatmaps.png)

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

![F_beh per slot and layer, three settings, full-state patch](https://raw.githubusercontent.com/superkaiba/explore-persona-space/94728abb5029fe49d9d632633f5a5e6b451b65d3/figures/issue_2094/userchat_heatmaps/f_beh_heatmaps.png)

**Takeaways:**

1. Matched query at the context vector moves behavior: F 0.43 at all 28 layers, 0.33 for the middle
   band, 0.29 at a single layer (L16) — a third to a half of a full context swap, never all of it.
2. Behavior transfer is unique to the context vector. Prefix-end reads ≈0 (−0.006 at all layers)
   *even though it moved the answer vector 0.18* — the prefix-end state reads the prefix out but does
   not causally carry it. 2nd/3rd-to-last are also flat.
3. Matched prefix is ≈0 at the context vector (−0.007). Query text reaches 0.92 at all layers, but on
   19/30 coherent draws — which query gets answered lives in the query's own token states, not v_C.
4. Cross transfers weakly (0.17 at all layers), consistent with Result 1.

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
(cells + per-strength values in `figures/issue_2094/userchat_heatmaps/dose_lineplot_summary.json`).
Per-cell tables `eval_results/issue_2094/f_metrics/{f_cells,null_cells,anchors}.jsonl`,
transport `eval_results/issue_2094/transport/`. Results 0–3 read at the full-state patch over all
non-degenerate coherent pairs; Result 4 over well-separated pairs (|anchor separation| ≥ 0.5) only.
