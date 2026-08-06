---
title: Context-end is the only single position where activation edits move behavior
  clear of the shuffled-donor null, and even its strongest patch falls well short
  of a full context swap (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-08-05T20:07:47Z'
has_clean_result: false
parent_id: 1415
origin_prompt: 'start in background with happy coder and setup periodic monitoring
  of it (design finalized in chat 2026-08-05: unified F metric, crossed 3-prefix x
  5-query bank, 3 settings, slot grid incl. prefix-end injection, greedy 1-draw grid
  + K=10 anchors, coherence>60 gating, fragility map, banked-map-only transport, stage-2
  best-cell confirmation at temp 1.0 K=5)'
workflow: v1
goal: 'On Qwen-2.5-7B-Instruct, test whether interventions (activation patching and
  steering) at single context positions — the context-end and prefix-end vectors,
  with last-3-token / query-span / full-context controls — causally move BOTH the
  answer state and behavior toward a target context, across three settings from one
  crossed context bank (matched query; matched prefix; cross), measured by one unified
  fraction-of-swap metric F at both levels (F_act signed projection with disjoint
  baseline halves; F_beh dual-judge contrast normalized between unpatched floor and
  generate-under-B ceiling), plus map-transport cosines against banked ridge maps
  only (context-end #779 963k L14/L19; prefix-end #1738 L14/L19/L26 — no new map training),
  an on-experiment linearity fit L (held out by pair; direction-aware vs banked M
  and the #1776 Jacobian J), coherence-gated reporting (judge > 60, coherent-only,
  <50%-coherent cells marked), and a fragility map vs the norm-matched shuffled-donor
  null.'
relates_to:
- spec-context-as-vector
- spec-steering
---
# Context-end is the only single position where activation edits move behavior clear of the shuffled-donor null, and even its strongest patch falls well short of a full context swap (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- A grid-wide bootstrap screen on well-separated pairs finds 36 of 1,245 behavior cell families with the steered mean above the shuffled-donor null on fully disjoint 95% intervals; 21 sit in cap-hit-compromised multi-token control cells and are untestable, and all 15 clean survivors are context-end edits — 14 matched-query (prefix transfer) and 1 cross — reading 0.18-0.63 of a full swap against nulls of -0.24 to 0.07.
- The largest clean effect is the full-state patch at all 28 layers of the context-end position: 0.63 of a full swap on prefix behavior (null -0.05, n = 10 pairs); clean single-layer edits at layers 12-20 read 0.18-0.33. The screen is post hoc and unadjusted across 2,172 comparisons, none of the 15 survivors was re-measured at stage 2, and the stage-1-selected winner cells themselves stay null-overlapping (0.20-0.22 vs nulls 0.04 to -0.02; p = 0.15-0.16, n = 10-12 pairs).
- Raw cell means of 0.85-2.39 — the stage-1 winners, with stage-2 confirmations at 1.17-1.86 — come from weakly-separated anchor pairs and collapse below 0.13 once pairs with anchor separation under 0.5 are excluded, a stratification chosen after seeing the results (the plan set no exclusion rule); at a 0.03-only exclusion the matched-query winner still reads 0.40.
- Prefix-end, second-to-last, and third-to-last slots yield zero null-separated behavior families anywhere on the grid; the prefix-end slot moves the answer state in raw terms (0.210) but its null matches (0.209). The activation read agrees with the behavior screen: 9 clean disjoint activation families, all at the same context-end cells (all-layer patch 0.41 vs null 0.09).
- The edit-to-response map is far from linear: shift magnitude is flat in dose (mean log-log slope 0.00-0.06 by slot-layer family vs 1.0 for a linear map), a one-operator fit reaches held-out R-squared 0.084 at best, and banked-map transport cosines top out at 0.16.
- Single-position edits keep fluency (0.7% incoherent vs 2.0% unpatched baseline); 16 steered cells breached the 2% cap-hit re-generation threshold unremediated — the query-span and last-three joint mid-stack controls (80% incoherent at the worst cell, null 77%) and context-end joint-stack cells — and those 16 cells hold the 21 untestable separating families. About 1% of grid rollouts flip into Chinese, arm-balanced; the 15 surviving cells carry zero intrusion flags (225 rows per arm).

## Goal

Test whether interventions at single context positions — the context-end and prefix-end summary states, with multi-token controls — causally move the answer representation and the generated behavior toward a target context, measured as a unified fraction-of-swap F at both levels across matched-query, matched-prefix, and cross pairs, with banked-map transport, linearity, and fragility reads alongside.

**This experiment in context:** the single-position steering conventions, the disjoint-baseline-halves activation read, and the injection-exactness gate come from [#1415](https://eps.superkaiba.com/tasks/1415); the frozen context-end ridge maps are [#779](https://eps.superkaiba.com/tasks/779)'s, the prefix-end maps [#1738](https://eps.superkaiba.com/tasks/1738)'s, and the layer-14-to-19 Jacobian [#1776](https://eps.superkaiba.com/tasks/1776)'s — transport only, no refitting.

**Broader narrative:** the mapping line shows the context-end state predicts the answer representation well when read passively; this experiment asks the causal converse by editing that state in place. Behavior follows only at context-end and only partially — even the maximal all-layer patch falls well short of a full swap — the response is far from linear in the edit, and the maps do not transport: passive predictability does not imply full single-position causal control.

## Methodology

**Design:** 15 contexts = 3 prefixes x 5 queries, verbatim in `src/explore_persona_space/experiments/issue2094/bank.py` — a plain default assistant, a strong pirate persona ("Captain Marrow"), and a constructed prior exchange (enthusiastic birthday-party planning) — giving 60 pairs: 30 matched-prefix (query transfer), 15 matched-query (prefix transfer), 15 stratified cross. Interventions install the pair difference (dose 0.5/1/2/4x) or the full target state (replace) at one position: context-end and prefix-end at all 28 single layers plus joint mid-stack (layers 14-20) and joint all-layer variants; second-to-last, third-to-last, last-three-joint, and query-span controls at joint mid-stack only. A prefix-centroid variant (query-averaged) runs at context-end on matched-query pairs. Every plotted cell has a norm-matched shuffled-donor null (seeded derangement, donor from a different pair). Floors and ceilings come from 10 unpatched temperature-1.0 draws per context; the grid is one greedy draw per cell (21,000 steered + 21,000 null); the 6 stage-1-best cells were re-measured at temperature 1.0 with 5 draws per pair, labeled post-selection. Matched-prefix pairs at prefix-end are degenerate by design (identical prefix states; 4,500 steered + 4,500 null rows flagged `degenerate_self` and excluded from aggregates). A follow-up analysis round on the existing rollouts (no new generation) added the grid-wide well-separated bootstrap recount, the banked-layer dose figure, and a corrected stage-1-vs-stage-2 join. Data realism: constructed tier-3 bank, a recorded design decision (strong-contrast prefixes chosen so ceilings separate) carried as a scope caveat.

**Training:** N/A — no model training.

| Generation / eval parameter | Value | Source |
|---|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct | run digest (`epm:results` reproducibility card) |
| `max_new_tokens` | 1024 | run digest |
| Grid decoding | greedy (temperature 0.0), 1 draw | run digest |
| Anchor / stage-2 decoding | temperature 1.0, 10 / 5 draws, seed 42 | run digest |
| Layers captured | all 28; activation read layer 26 (span-mean answer vector) | plan §4.4 / `f_cells.jsonl` `read_layer` |
| Judge | claude-sonnet-4-5-20250929, graded 0-100, 1 draw per rubric, `max_tokens` 1024, Batch API | `judge_summary.json` `instrument` |
| Coherence gate | form-only rubric, coherent = score > 60 | plan §4.5 / `coherence_baseline_gate.json` |
| One-operator ridge fit | top-128-PC subspace, 240 observations per family, grouped folds by pair (10) and by context family (15) | `l_fit_results.json` |
| Bootstrap | pair-clustered, B = 10,000, seed 20941; same settings for the well-separated recount | `bootstrap_cis.json`, `bootstrap_cis_wellsep.json` |
| Bank / donor-derangement seed | 2094 | run digest |

**Evaluation:** behavior F per draw = (judge contrast of the rollout minus the floor mean) / (ceiling mean minus floor mean); judge contrast = (score against descriptor B minus score against descriptor A)/100; matched-prefix pairs use query rubrics, matched-query pairs prefix rubrics, cross pairs both. Activation F = signed projection of the patched-minus-floor answer-state shift onto the ceiling-minus-floor axis, floor split into disjoint halves. All reported quantities use coherent draws only. "Well-separated" below means absolute anchor separation of at least 0.5 (judge-contrast units; maximum 2). Anchor separations on the prefix rubric: 5 of 15 matched-query and 3 of 15 cross pairs fall below 0.5, of which 3 and 2 sit at 0.03 or less, and one retained cross pair is inverted at -0.96 (its ceiling judged below its floor); all matched-prefix and cross query-rubric separations are at least 0.5. The plan specified per-pair separation reporting with no exclusion rule; the well-separated stratification in Results was chosen after seeing the data, not written into the plan. The grid-wide screen recounts the pair-clustered bootstrap under that restriction with the parent run's own batched implementation, exclusions, and gating: per cell family (setting x slot x layer variant x dose x edit-direction type x metric), steered and null means with 95% intervals over well-separated pairs — 2,172 steered-vs-null comparisons at five or more pairs (927 activation, 757 prefix-rubric, 488 query-rubric behavior). "Separates" in Results means fully disjoint 95% intervals with steered above (4 further behavior families are disjoint with steered below); the weaker read — null mean outside the steered interval — fires on 223 prefix-rubric, 38 query-rubric, and 117 activation families in either direction and carries no claims. The screen is post hoc, with no multiplicity adjustment across families. 148,650 judge calls in 27 Batch-API waves: 25 content drops, 0 residual transport losses, 0 truncation drops; the judge pilot gate, the anchor coherence gate (median 100, 98% of draws above 60), and the injection-exactness gate (12 of 12 spot cells) all passed before production. Paired steered-vs-null p-values in Results are exact signed-rank tests over pairs; steered-vs-null interval comparisons use the pair-clustered bootstrap. Cell-level agreement between the two F levels: Spearman rho = 0.43-0.50 across 929 steered cell families (p < 1e-40), so the continuous activation companion tracks the judged behavioral read. Mechanical audits (script intrusion at non-Latin letter fraction > 0.05, repetition, empty output) ran on every arm: grid 214 of 21,000 steered vs 239 of 21,000 null flagged; anchors 7 of 150; stage-2 31 of 600; intrusion-excluded recounts of every stage-2 cell shift coherence-gated means by at most 0.02, and the 15 surviving screen cells carry zero intrusion flags in either arm (225 rows each).

**Data extraction:** per-context state vectors captured in one right-padded forward pass per context (positions read off token ids, never re-tokenized strings); answer vectors are span-means over the model's own completion at all 28 layers, in both the parent convention and the tail-inclusive pooling used for map transport (parity of each banked map's input convention with the injected slot recorded in `eval_results/issue_2094/map_parity.json` before any transport number). Every F table row carries pair, slot, layer variant, dose, coherence flag, cap-hit flag, and donor pair id.

Acknowledged conciseness overages (deliberate, WARN-tier): several Takeaways bullets exceed 30 words and five Results blocks exceed 120 words — the denominator-artifact dissection, the non-linearity battery, the breach disclosures, and the grid-wide screen's reconciliation each need the extra clauses to stay verifiable against the artifacts; total content prose runs over the 800-word budget for the same reason.

**Sample training/evaluation data + completions:** all sample blocks below are cherry-picked from the best-behavior stage-2 cell (matched-query, context-end, layer 19, dose 4x, prefix-centroid; temperature 1.0), selected as the highest and lowest per-draw behavior F on well-separated pairs — not a random sample; all 600 stage-2 rows: [stage-2 rollouts @ pinned revision](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/cfedd60af9e061e9efa42a1573ebeab9ec790eca/issue2094_singlepos/raw_completions/stage2/s2_s2__matched_query__ce__L19__a4__B.jsonl).

Cherry-picked: the three highest per-draw F ("firing") draws (not a random sample); all rows: [same file](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/cfedd60af9e061e9efa42a1573ebeab9ec790eca/issue2094_singlepos/raw_completions/stage2/s2_s2__matched_query__ce__L19__a4__B.jsonl).

<details>
<summary>Three high-F draws — the source register is lost; the target register is acquired only sometimes</summary>

Plain-assistant context, hash-table query, pirate-centroid edit (F = 0.58): "yarrrr! Matey, aye, let's biket'bout hash tables, fer be! A hash table is quite the nifty tool in yer book of technologies…" — genuine pirate-register acquisition.

Pirate context, party-conversation target, story query (F = 0.94): the draw answers entirely in Chinese (a ghost story about the lighthouse) — the high F comes from losing the pirate register, not from acquiring the target's enthusiastic register.

Pirate context, party-conversation target, rent-vs-buy query (F = 0.60): "MainForme! When it comes to picking 'twixt rentin' 'n buyin' a crib…" — pirate register partly diluted, target register absent.

</details>

Cherry-picked: the three lowest per-draw F ("non-firing") draws (not a random sample); all rows: [same file](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/cfedd60af9e061e9efa42a1573ebeab9ec790eca/issue2094_singlepos/raw_completions/stage2/s2_s2__matched_query__ce__L19__a4__B.jsonl).

<details>
<summary>Three low-F draws — the edit leaves the source register intact</summary>

Pirate context, party-conversation target, interview query (F = -0.41): "Certainly, lad, preparin' fer yer first job interview is like settin' sails for a stormy sea…" — fully pirate.

Pirate context, party-conversation target, hash-table query (F = -0.26): "Certainly, me boy. Ye've asked 'bout hash tables, an' well, they be like a magical treasure chest on me frigate!" — fully pirate.

Plain context, pirate target, story query (F = -0.35): "me (Qwen) am a large language model and do not have personal experiences, but I can craft a beautiful narrative for you…" — plain register with a trace of pirate grammar, scored non-pirate by the judge.

</details>

For reference, a generate-under-target ceiling draw (pirate context, hash-table query): "Ahoy there, lippy! To explain 'bout hash tables, ye need to know first that they're like a big ol' treasure chest…" — all 150 anchor draws: [anchors @ pinned revision](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/cfedd60af9e061e9efa42a1573ebeab9ec790eca/issue2094_singlepos/raw_completions/anchors/anchors.jsonl).

## Results

### Behavior barely moves at the stage-1-selected cells once anchors separate

The heatmap maps mean behavior F per steered cell (rows = slots, columns = steered layer, dose facets, one panel per setting).

![Behavior fraction-of-swap heatmap across slots, layers, doses, and settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/hero2_f_beh_heatmap.png)

> **Figure.** *Nearly every single-layer cell reads F near zero.* Circles mark degenerate-by-design cells, x marks cells under 50% coherent; the saturated cells at joint variants and late context-end layers are the weak-anchor artifacts dissected in the next section.

On well-separated pairs the stage-1-selected best cells read 0.22 (cross, context-end layer 14, dose 4x; null 0.04; p = 0.16, n = 12) and 0.20 (matched query, context-end layer 14, full-state patch; null -0.02; p = 0.15, n = 10) — not separated from the shuffled-donor null at these cells, though the grid-wide screen in the final section separates neighboring context-end cells cleanly.

The cross cell's well-separated mean replicates at stage 2 (0.22 greedy, 0.21 at temperature 1.0, 5 draws per pair). Both matched-prefix selections confirm at 0.02 or less: no query transfer at the selected cells. Script intrusion is arm-balanced, shifting no coherence-gated stage-2 cell mean by more than 0.02.

### The 0.85-2.39 cell means come from weakly-separated anchor pairs and collapse when those pairs are excluded

The figure recounts the four largest stage-1 cells: all-pairs means beside well-separated-only means, steered and null, with per-pair points.

![Separation-stratified behavior F recount at the four largest cells with per-pair points](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/result_sep_stratified_fbeh.png)

> **Figure.** *The 0.85-2.39 cell means collapse to 0.22 or less on well-separated pairs.* Aggregate + per-pair companion pair: left, means with pair-clustered bootstrap error bars (B = 10,000; n = 10-15 pairs per bar); right, per-pair F — steered and null interleave.

Weak anchor denominators inflate cell means 30-200 fold at separations of 0.03 or less; every weakly-separated pair sits in the matched-query and cross prefix rubrics (counts in Methodology). Raw winner means reach 0.85-2.39 (nulls to 5.5).

The stratification was chosen after seeing the results — the plan called for per-pair reporting with no exclusion — and it is threshold-dependent: dropping only pairs at 0.03 or less leaves the matched-query winner at 0.40; the collapse to 0.006 needs the 0.5 bar (the cross winner collapses under either, 0.07). On well-separated pairs both stage-1 winners read below their own nulls (0.006 vs 0.051; 0.077 vs 0.176), and stage-2 confirmations collapse the same way — 1.86 to 0.12, 1.17 to 0.09.

One retained cross pair carries an inverted anchor (-0.96, ceiling judged below floor); excluding it moves the best cross cell from 0.22 to 0.24.

### Activation-level movement is small at context-end and non-specific at prefix-end

The heatmap shows mean activation F per cell — the signed projection of the answer-state shift onto the ceiling-minus-floor axis at read layer 26 — in the same layout as the behavior heatmap.

![Activation fraction-of-swap heatmap across slots, layers, doses, and settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/hero1_f_act_heatmap.png)

> **Figure.** *Matched-query edits move the answer state at both end slots; only the context-end movement is pair-specific.* Mean activation F per cell (disjoint floor halves; n = 15-30 pairs); matched-prefix and cross panels sit near zero; the saturated query-span cells are the fragility cells of the next section.

Matched-query context-end edits move the answer state by a small, real amount: 0.16 steered vs 0.09 null at layer 14, dose 1x — the largest steered-minus-null excess on the all-pairs grid — with overlapping bootstrap intervals (0.09 to 0.23 vs 0.02 to 0.17). The prefix-end slot moves more in raw terms but fails specificity outright: its best raw cell (layer 26, full-state patch) reads 0.210 steered vs 0.209 null. Under the well-separated restriction the activation read separates cleanly at nine context-end cells — the all-layer patch (0.41 vs null 0.09) and mid-stack joint edit at half dose (0.27 vs 0.03) lead — the same cells the behavior screen keeps.

### The response to an injected edit is strongly non-linear, and banked maps do not transport

The figure shows the cosine between the realized answer-state shift and each banked map's predicted shift, per map and dose regime, steered vs shuffled-donor null, with per pair-dose points (tail-inclusive pooling for map parity).

![Banked-map transport cosines at context-end and prefix-end cells, steered vs null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/result1b_transport_cosines.png)

> **Figure.** *Banked maps predict almost none of the realized shift.* Bars are per-map means pooled over additive doses (left panel) and the full-state patch (right panel), steered vs null; points are pair-by-dose values. Pooled bars top 0.09; the best per-dose cell mean is 0.16 (context-end layer-14 map, dose 2x, prefix-centroid); the full-state context-map null matches its steered bar.

The realized shift's magnitude barely grows over an 8x dose range (mean log-log slope 0.00-0.06 by slot-layer family vs 1.0 for a linear map; the prefix-end families are fully dose-insensitive, slope 0.00 with median adjacent-dose cosine 1.00) and its direction drifts across doses at context-end (median adjacent-dose cosine 0.62 at layer 14; median split-half reliability 0.93, so the drift is not noise). The one-operator fit explains at most 8.4% of held-out response variance (2.3% family-held-out) and does not align with the banked map at layer 14 (Procrustes cosine 0.256 vs rotation-null 97.5th percentile 0.293). The additivity spot-check is weak where informative (cosine 0.17-0.88); its reported 1.000 comes from a combo whose three rollouts were the same greedy completion — insensitivity, not additivity.

### Single-position edits are safe; multi-position mid-stack edits break fluency non-specifically

The heatmaps show excess incoherence per cell (incoherent fraction minus the 2.0% unpatched anchor baseline) for steered and null arms, with the steered cap-hit fraction as a companion panel.

![Fragility heatmaps of excess incoherence for steered and null arms with cap-hit companion](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/result4_fragility.png)

> **Figure.** *Both arms stay coherent everywhere except query-span and last-three joint mid-stack cells.* Excess incoherence per slot x layer-variant x dose (n = 60-75 rollouts per cell arm), plus the steered cap-hit fraction.

Steered incoherence totals 0.7% (149 of 21,000) against the 2.0% unpatched baseline — single-position edits, full-state patches included, essentially never break fluency. The fragile cells are the multi-position controls at joint mid-stack layers: query-span at dose 4x reaches 80% incoherent (null 77%) and 73% cap-hit — generic disruption the wrong-pair null reproduces. Sixteen steered cells breached the plan's 2% cap-hit re-generation threshold unremediated: the query-span and last-three families (13-46% pooled by slot and setting over doses and arms; 5-73% per cell; last-three matched-prefix, 0.3%, did not breach) and the context-end joint-stack cells (2.7-6.7%) — the same cells that saturate the behavior heatmap — so their F values are untestable rather than transfer evidence, though several truncated query-span rollouts visibly moved query content while destroying fluency, the only sighting of query transfer in the grid.

### A grid-wide screen on well-separated pairs finds null-separated transfer only at context-end

The figure shows behavior F dose response at the two banked layers (14 and 19) on well-separated pairs: steered vs shuffled-donor null means with pair-clustered bootstrap bands (pair-difference edits), per-pair traces behind, one column per setting, context-end and prefix-end rows.

![Behavior F dose response at layers 14 and 19 on well-separated pairs, steered vs shuffled-donor null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/267387d2c9b448d3d0f29a1cb7bc8e43cd222e36/figures/issue_2094/result_fbeh_dose_L14L19_wellsep.png)

> **Figure.** *The steered band clears the null only in the matched-query column at context-end.* Dose response on well-separated pairs (n = 10-12 pairs per band; bands: pair-clustered bootstrap, B = 10,000); matched-prefix by prefix-end panels are degenerate by design.

Grid-wide, 36 of 1,245 behavior cell families read steered above null with fully disjoint 95% intervals, but 21 sit in the cap-hit-breached cells of the previous section, leaving 15 clean survivors — all context-end, 14 matched-query and 1 cross, all on the prefix rubric. Clean single layers 12-20 read 0.18-0.33 (nulls -0.24 to -0.00), the mid-stack joint edit at half dose 0.43, and the all-layer full-state patch 0.63 vs null -0.05 (n = 10 pairs).

Prefix-end and the single-token controls: zero. The screen is post hoc and unadjusted; each cell is one greedy draw per pair; none of the 15 was stage-2 re-measured; the surviving cells carry zero intrusion flags.

---

**Repro:** pod run at code SHA `cf7f2254ce` (grid + anchors, 1x H100, 42,000 rollouts, cap-hit 0.69%; stage-2 pod `pod-2094-s2`, 600 + 18 rollouts, cap-hit 0); off-pod analysis + figures at `5552c3e3c7`…`9d252d52c3` on branch `issue-2094`; follow-up analysis round (grid-wide well-separated bootstrap, banked-layer dose figure, corrected stage-1-vs-stage-2 join) at `267387d2c9`, code-review PASS, no new generation. Figures: `figures/issue_2094/` (18 run figures + `result_sep_stratified_fbeh` via `scripts/issue2094_sep_stratified_fig.py` + the follow-up round's two); linked-not-embedded (deliberate): [per-pair anchor separations](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/exp_anchor_separation.png), [activation dose curves](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/hero1_f_act_dose_curves.png), [stage-1 vs stage-2 selected-cell scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/267387d2c9b448d3d0f29a1cb7bc8e43cd222e36/figures/issue_2094/exp_stage2_vs_stage1.png) (re-rendered in the follow-up round; the run's original render mis-joined the stage-1 cells and drew a fallback axis). Eval tables: `eval_results/issue_2094/` (`f_metrics/f_cells.jsonl`, `null_cells.jsonl`, `anchors.jsonl`, `bootstrap_cis.json`, `bootstrap_cis_wellsep.json`; `linearity/`, `transport/`, `fragility/`, `judge/`, `gates/`, `best_cells.json`, `map_parity.json`). HF (verified live via `list_repo_tree`): [issue2094_singlepos](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/cfedd60af9e061e9efa42a1573ebeab9ec790eca/issue2094_singlepos) — `raw_completions/{grid,anchors,stage2,stage2_additivity,judge_raw}` @ cfedd60af9e061e9efa42a1573ebeab9ec790eca (880 grid shards) plus analysis tensors (`va_store`, `vc_bank`, stage-2 stores). Reused artifacts: ridge maps from [#779](https://eps.superkaiba.com/tasks/779) (`issue779_monitoring/n1m_readout/weights/{L14,L19,L26}/ridge.pt` @ cfedd60af9e061e9efa42a1573ebeab9ec790eca) and [#1738](https://eps.superkaiba.com/tasks/1738) (`issue1738_multiturn/analysis_tensors/weights/{L14,L19,L26}/prefix_ridge.pt` @ cfedd60af9e061e9efa42a1573ebeab9ec790eca) — fit: transport-only frozen instruments, input-convention parity recorded in `map_parity.json`; Jacobian `J_last.pt` from [#1776](https://eps.superkaiba.com/tasks/1776). All behavioral reads are on-policy generations (greedy grid; temperature-1.0 anchors and stage-2); no teacher forcing. Known deviations (all recorded in the run digest): per-position add realization of multi-position replace doses; retokenized cap-hit basis; the null installs the donor's target state (not a difference) at replace cells; matched-prefix prefix-end degeneracy.

**Context:** originating prompt (verbatim): "start in background with happy coder and setup periodic monitoring of it (design finalized in chat 2026-08-05: unified F metric, crossed 3-prefix x 5-query bank, 3 settings, slot grid incl. prefix-end injection, greedy 1-draw grid + K=10 anchors, coherence>60 gating, fragility map, banked-map-only transport, stage-2 best-cell confirmation at temp 1.0 K=5)". Design finalized interactively in chat 2026-08-05; run 2026-08-06 (grid pod ~4 h; judge waves + stage-2 same day); free-analysis follow-up round folded 2026-08-06. Lineage: steering conventions from [#1415](https://eps.superkaiba.com/tasks/1415); banked maps [#779](https://eps.superkaiba.com/tasks/779)/[#1738](https://eps.superkaiba.com/tasks/1738); Jacobian [#1776](https://eps.superkaiba.com/tasks/1776).
