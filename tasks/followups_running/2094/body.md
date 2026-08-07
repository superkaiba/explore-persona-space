---
title: Context-end is the only single position where activation edits move behavior
  clear of the shuffled-donor null, and even its strongest patch falls well short
  of a full context swap (MODERATE confidence)
kind: experiment
tags:
- followup-manual
created_at: '2026-08-05T20:07:47Z'
has_clean_result: true
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

**Methodology:** [docs/methodology/issue_2094.md](https://github.com/superkaiba/explore-persona-space/blob/e3f9e2d62c8d2b7245f87e6b9f0b4fee3f1b3510/docs/methodology/issue_2094.md) · [gist mirror](https://gist.github.com/superkaiba/ae1636afdacfa2da12970e5a0a20e488)

## Takeaways

- A grid-wide bootstrap screen on well-separated pairs finds 36 of 1,245 behavior cell families with the steered mean above the shuffled-donor null on fully disjoint 95% intervals; 21 sit in cap-hit-compromised cells — 15 in the multi-token controls, 6 at context-end joint-layer cells — and are untestable, and all 15 clean survivors are context-end edits — 14 matched-query (prefix transfer) and 1 cross — reading 0.18-0.63 of a full swap against nulls of -0.24 to 0.07.
- The largest clean effect is the full-state patch at all 28 layers of the context-end position: 0.63 of a full swap on the greedy grid and 0.51 under re-sampling (null 0.10); clean single-layer edits at layers 12-20 read 0.18-0.33. Independent re-sampling at temperature 1.0 (5 draws per pair) confirms 10 of 15 survivors on fully disjoint 95% intervals (steered 0.17-0.51, nulls -0.05 to 0.10); the other 5 stay steered-above with overlapping intervals. The screen is post hoc across 2,172 comparisons, the confirmation is post-selection, and the stage-1 winner cells stay null-overlapping (0.20-0.22; p = 0.15-0.16).
- Raw cell means of 0.85-2.39 — the stage-1 winners, with stage-2 confirmations at 1.17-1.86 — come from weakly-separated anchor pairs and collapse below 0.13 once pairs with anchor separation under 0.5 are excluded, a stratification chosen after seeing the results (the plan set no exclusion rule); at a 0.03-only exclusion the matched-query winner still reads 0.40.
- Prefix-end, second-to-last, and third-to-last slots yield zero null-separated behavior families anywhere on the grid, and the activation read agrees: nine clean disjoint activation families, all at context-end. A follow-up round shows chat-template tokens were what made the query span untestable — editing only the query's text tokens yields 14 of 70 clean family reads — the 8 query-rubric behavior reads separate at 0.68-0.95 of a full swap against elevated nulls of 0.22-0.56, all 14 at half dose or the mid-stack joint band — where the template-inclusive query span has zero and context-end has four.
- The edit-to-response map is far from linear: shift magnitude is flat in dose (mean log-log slope 0.00-0.06 by slot-layer family vs 1.0 for a linear map), a one-operator fit reaches held-out R-squared 0.084 at best, and banked-map transport cosines top out at 0.16.
- Single-position edits keep fluency (0.7% incoherent vs 2.0% unpatched baseline); whole-span patches break it in both arms — the follow-up's template-inclusive prefix patches run 93-97% judge-incoherent and 25 of the round's 30 pooled cells breach the 2% cap-hit trigger at a 2048 cap — so whether prefix information is distributed across its span stays unmeasured, not answered in the negative. Re-generating the parent's 16 breached cells at 2048 moved no well-separated mean by more than 0.023; the 15 surviving cells and the follow-up's four clean query-text cells carry zero script-intrusion flags.

## Goal

Test whether interventions at single context positions — the context-end and prefix-end summary states, with multi-token controls — causally move the answer representation and the generated behavior toward a target context, measured as a unified fraction-of-swap F at both levels across matched-query, matched-prefix, and cross pairs, with banked-map transport, linearity, and fragility reads alongside.

**This experiment in context:** the single-position steering conventions, the disjoint-baseline-halves activation read, and the injection-exactness gate come from [#1415](https://eps.superkaiba.com/tasks/1415); the frozen context-end ridge maps are [#779](https://eps.superkaiba.com/tasks/779)'s, the prefix-end maps [#1738](https://eps.superkaiba.com/tasks/1738)'s, and the layer-14-to-19 Jacobian [#1776](https://eps.superkaiba.com/tasks/1776)'s — transport only, no refitting.

**Broader narrative:** the mapping line shows the context-end state predicts the answer representation well when read passively; this experiment asks the causal converse by editing that state in place. Behavior follows only at context-end and only partially — even the maximal all-layer patch falls well short of a full swap — the response is far from linear in the edit, and the maps do not transport: passive predictability does not imply full single-position causal control.

## Methodology

**Design:** 15 contexts = 3 prefixes x 5 queries, verbatim in `src/explore_persona_space/experiments/issue2094/bank.py` — a plain default assistant, a strong pirate persona ("Captain Marrow"), and a constructed prior exchange (enthusiastic birthday-party planning) — giving 60 pairs: 30 matched-prefix (query transfer), 15 matched-query (prefix transfer), 15 stratified cross. Interventions install the pair difference (dose 0.5/1/2/4x) or the full target state (replace) at one position: context-end and prefix-end at all 28 single layers plus joint mid-stack (layers 14-20) and joint all-layer variants; second-to-last, third-to-last, last-three-joint, and query-span controls at joint mid-stack only. A prefix-centroid variant (query-averaged) runs at context-end on matched-query pairs. Every plotted cell has a norm-matched shuffled-donor null (seeded derangement, donor from a different pair). Floors and ceilings come from 10 unpatched temperature-1.0 draws per context; the grid is one greedy draw per cell (21,000 steered + 21,000 null); the 6 stage-1-best cells were re-measured at temperature 1.0 with 5 draws per pair, labeled post-selection. Matched-prefix pairs at prefix-end are degenerate by design (identical prefix states; 4,500 steered + 4,500 null rows flagged `degenerate_self` and excluded from aggregates). A follow-up analysis round on the existing rollouts (no new generation) added the grid-wide well-separated bootstrap recount, the banked-layer dose figure, and a corrected stage-1-vs-stage-2 join. A second, GPU-backed follow-up round (`fu1_regen_confirm`) executed the plan's cap-hit re-generation trigger — the 16 breached pooled cells re-generated in both arms with the run block otherwise verbatim and `max_new_tokens` raised from 1024 to 2048 (2,100 of the grid's 42,000 rollouts) — and independently re-measured the 15 surviving screen families at temperature 1.0 with 5 draws per pair, steered and shuffled-donor-null arms under the parent's seeded derangement; all follow-up generations are on-policy (greedy re-generation, temperature-1.0 confirmation draws). A third, user-requested follow-up round (`fu2_span_slots`) extended the intervention grid with three span slots on the same instrument — the final user turn's query text tokens with chat-template tokens excluded, the whole prefix span including template tokens, and the prefix content tokens only — at joint all-layer and joint mid-stack variants, the full dose ladder plus full-state patch, pair-difference edits with a norm-matched shuffled-donor null per cell: 30 families / 60 blocks / 2,400 greedy cells at `max_new_tokens` 2048 from the start, on a fresh pod whose state bank passed a capture-parity gate against the parent bank (early-layer minimum cosine 0.9999999, flattened all-layer read 1.0). Template tokens are identified structurally from the tokenized chat-template ids with boundary asserts, never by matching decoded text; prefix-span slots exclude matched-prefix pairs by design (identical prefixes make the pair delta zero). Edits right-align to the pair's minimum overlap within each span's own coordinate set (content tokens for the two text-only spans); the template-inclusive prefix span right-aligns within the full span, so across prefixes of unequal length the aligned offsets can pair template with content positions — a scope caveat the content-only span does not carry. Cap-hit is reporting-only in this round: compromised family reads are computed and labeled, never dropped or re-generated. Data realism: constructed tier-3 bank, a recorded design decision (strong-contrast prefixes chosen so ceilings separate) carried as a scope caveat.

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
| Follow-up re-generation (round `fu1_regen_confirm`) | 16 breached cells, both arms, greedy, `max_new_tokens` 2048, run block otherwise verbatim | `f_metrics/fu1/fu1_regen_swap.json` |
| Follow-up confirmation (round `fu1_regen_confirm`) | 15 survivor cells, temperature 1.0, 5 draws per pair (per-draw seeds), steered + shuffled-donor null | `f_metrics/fu1/fu1_conf1_confirmation.json` |
| Follow-up judge | production instrument unchanged, 18 Batch-API waves, 14,310 items, isolated work root | `judge_fu1/judge_summary.json` |
| Follow-up grid (round `fu2_span_slots`) | 3 span slots x 2 joint layer variants x dose ladder + full-state patch, 30 families / 2,400 cells, greedy, `max_new_tokens` 2048 | `scripts/issue2094_fu2.py` + `f_metrics/fu2/fu2_summary.json` |
| Follow-up verdict gate (round `fu2_span_slots`) | disjoint 95% pair-clustered bootstrap intervals (B = 10,000, seed 20941), at least 5 well-separated pairs, pooled steered cap-hit at most 2% | `fu2_summary.json` `verdict_definition` |
| Follow-up judge (round `fu2_span_slots`) | production instrument unchanged, 9 Batch-API waves, 9,000 items, isolated work root | `judge_fu2/judge_summary.json` |

**Evaluation:** behavior F per draw = (judge contrast of the rollout minus the floor mean) / (ceiling mean minus floor mean); judge contrast = (score against descriptor B minus score against descriptor A)/100; matched-prefix pairs use query rubrics, matched-query pairs prefix rubrics, cross pairs both. Activation F = signed projection of the patched-minus-floor answer-state shift onto the ceiling-minus-floor axis, floor split into disjoint halves. All reported quantities use coherent draws only. "Well-separated" below means absolute anchor separation of at least 0.5 (judge-contrast units; maximum 2). Anchor separations on the prefix rubric: 5 of 15 matched-query and 3 of 15 cross pairs fall below 0.5, of which 3 and 2 sit at 0.03 or less, and one retained cross pair is inverted at -0.96 (its ceiling judged below its floor); all matched-prefix and cross query-rubric separations are at least 0.5. The plan specified per-pair separation reporting with no exclusion rule; the well-separated stratification in Results was chosen after seeing the data, not written into the plan. The grid-wide screen recounts the pair-clustered bootstrap under that restriction with the parent run's own batched implementation, exclusions, and gating: per cell family (setting x slot x layer variant x dose x edit-direction type x metric), steered and null means with 95% intervals over well-separated pairs — 2,172 steered-vs-null comparisons at five or more pairs (927 activation, 757 prefix-rubric, 488 query-rubric behavior). "Separates" in Results means fully disjoint 95% intervals with steered above (4 further behavior families are disjoint with steered below, all four at prefix-end); the weaker read — null mean outside the steered interval — fires on 223 prefix-rubric, 38 query-rubric, and 117 activation families in either direction and carries no claims. The screen is post hoc, with no multiplicity adjustment across families. 148,650 judge calls in 27 Batch-API waves: 25 content drops, 0 residual transport losses, 0 truncation drops; the judge pilot gate, the anchor coherence gate (median 100, 98% of draws above 60), and the injection-exactness gate (12 of 12 spot cells) all passed before production. Paired steered-vs-null p-values in Results are exact signed-rank tests over pairs; steered-vs-null interval comparisons use the pair-clustered bootstrap. Cell-level agreement between the two F levels: Spearman rho = 0.43-0.50 across 929 steered cell families (p < 1e-40), so the continuous activation companion tracks the judged behavioral read. Mechanical audits (script intrusion at non-Latin letter fraction > 0.05, repetition, empty output) ran on every arm: grid 214 of 21,000 steered vs 239 of 21,000 null flagged for script intrusion (repetition: 111 vs 96; none empty); anchors 7 of 150; stage-2 31 of 600 (both intrusion-only); intrusion-excluded recounts of every stage-2 cell shift coherence-gated means by at most 0.02, and the 15 surviving screen cells carry zero intrusion flags in either arm (225 rows each). The follow-up round's judge ran the production instrument unchanged in an isolated work root: 18 of 18 Batch-API waves complete, 14,310 judged items, 15 content drops (0.10%), zero truncation drops, zero residual transport losses. Re-generated rows were reduced under the parent conventions and substituted into the well-separated bootstrap; the 4,112 families untouched by the swap reproduce the parent artifact exactly, and the swapped cells' well-separated means move by at most 0.023 (median 0.000). Confirmation verdicts use the same pair-clustered bootstrap (B = 10,000, seed 20941) on each family's own rubric over well-separated pairs; 34 of 1,520 confirmation draws were incoherent-excluded, none judge-dropped. The re-generated cells' activation tensors were persisted but not re-analyzed; the parent artifact remains the activation record at those cells. The span follow-up round reuses the same instrument end to end and classifies each of its 170 family reads (70 query-text, 50 per prefix span; a family read = setting x span x layer variant x dose x metric) with the same pair-clustered bootstrap: clean separation means fully disjoint 95% intervals with steered above on well-separated pairs, at least 5 usable pairs, and pooled steered cap-hit of at most 2%; a separating family over that cap-hit bar is labeled compromised rather than dropped; a family where either arm lacks scoreable coherent pairs is not comparable; incoherent and judge-dropped draws are counted per family, never coerced (per-family quality counts in `fu2_f_reads.json`). Parent comparables are recounted from the parent grid's own well-separated bootstrap artifact restricted to the follow-up-comparable subset (query span and context-end, pair-difference edits, joint layer variants), carrying the earlier round's 1024-cap breach flags. The round's judge ran the production instrument unchanged in an isolated work root: 9 of 9 Batch-API waves complete, 9,000 items, 8,980 scored, 20 content drops (0.22%), zero truncation drops, zero residual transport losses. Script-intrusion audit over all 2,400 rollouts of the round: the four cells carrying the 14 clean query-text reads have zero intruded rows in either arm (0 of 480 rows), so intrusion-zeroed and intrusion-excluded recounts leave the clean set unchanged; pooled query-text intrusion (31 of 600 steered vs 13 of 600 null rows) concentrates in the cap-hit-compromised high-dose cells, and the prefix-span slots' small coherent pools carry at most one intruded row per arm.

**Data extraction:** per-context state vectors captured in one right-padded forward pass per context (positions read off token ids, never re-tokenized strings); answer vectors are span-means over the model's own completion at all 28 layers, in both the parent convention and the tail-inclusive pooling used for map transport (parity of each banked map's input convention with the injected slot recorded in `eval_results/issue_2094/map_parity.json` before any transport number). Every F table row carries pair, slot, layer variant, dose, coherence flag, cap-hit flag, and donor pair id.

Acknowledged conciseness overages (deliberate, WARN-tier): several Takeaways bullets exceed 30 words and eight Results blocks exceed 120 words — the denominator-artifact dissection, the non-linearity battery, the breach disclosures, the grid-wide screen's reconciliation, the follow-up round's dual stress test, and the span round's verdict accounting each need the extra clauses to stay verifiable against the artifacts; total content prose runs over the 800-word budget for the same reason.

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

Cherry-picked from the follow-up confirmation round: the three highest and three lowest per-draw F at the largest confirmed cell (matched query, context-end, all-layer full-state patch; temperature 1.0) — not a random sample; all rows: [confirmation rollouts @ pinned revision](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/ce5a712423fb9bd8c3c4bda91c4886d28865d643/issue2094_singlepos/raw_completions/fu1_regen_confirm/conf1/fu1_fu1__matched_query__ce__joint_all__replace__A__steered.jsonl).

<details>
<summary>Follow-up confirmation draws at the all-layer patch — the same acquisition-vs-loss pattern as stage 2</summary>

Plain context, pirate target, hash-table query (F = 1.00): "Arrr, listen, ye fella! Ye be talkin' 'bout hash tables, fer sho'. Now a hash table, it be a mighty tool in the programmer's arsenal…" — genuine pirate-register acquisition.

Pirate context, party-conversation target, story query (F = 0.94): "The salty wind whispered secrets as it danced around the skeletal bones of the lighthouse…" — plain narrative; the high F comes from losing the pirate register, not acquiring the target's.

Pirate context, party-conversation target, story query (F = 0.94): "The salty tang of the Atlantic was always in his nose, even when he sat hunched over the ancient wooden desk in the lantern room…" — the same loss-not-acquisition pattern.

Pirate context, party-conversation target, hash-table query (F = -0.07): "Sure lassie, let me breakup this tech talk into a more shipboard friendly manner! Ahoy, ye scrub! A hash table is like a toolbox…" — fully pirate.

Plain context, pirate target, story query (F = -0.04): "Ainley Point, with its jagged cliffs and endless stretch of grey-tinted sea, was home to just one soul: Daniel Preston, the lighthouse keeper…" — plain register, no pirate.

Plain context, pirate target, story query (F = -0.04): "Ah, the whisper of wind through the tangle of pine needles, the steady pulse of the ocean's breath below…" — plain register, no pirate.

</details>

Cherry-picked from the span follow-up round: the three highest and three lowest per-draw query-rubric F among coherent steered draws at the strongest clean query-text cell (matched prefix, all 28 layers, half dose), plus one high-F shuffled-donor null draw — not a random sample; all rows: [steered rollouts @ pinned revision](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/06ddddb7c1352d58917f4710cb0670bfc6024a22/issue2094_singlepos/raw_completions/fu2_span_slots/rollouts/shard_qtext__joint_all__a0p5__A__steered.jsonl) and [null rollouts @ same revision](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/06ddddb7c1352d58917f4710cb0670bfc6024a22/issue2094_singlepos/raw_completions/fu2_span_slots/rollouts/shard_qtext__joint_all__a0p5__A__null.jsonl).

<details>
<summary>Query-text edits at half dose swap the answered query while the prefix register survives</summary>

Pirate prefix, sky-color query edited toward the lighthouse-story query (F = 1.05): "Twas a stormy night, an' the waves were as high as the keeper’s own hopes for a safe passage. He stood atop the stone walls of his solitary home…" — the target query is answered in the source prefix's register.

Pirate prefix, sky-color query edited toward the rent-vs-buy query (F = 1.02): "Ah, matey, that's a tricky question, better to be a captain of the high seas than a mere tenant on dry land! Rentin' or rentin'... well, it all depends on yer situation…"

Plain prefix, sky-color query edited toward the lighthouse-story query (F = 1.01): "In the misty embrace of the Atlantic, where the cold sea air whispered secrets and the waves sang lullabies to the night, stood the solitary beacon of Hope’s Rock Lighthouse…"

Party-planning prefix, interview query edited toward the rent-vs-buy query (F = 0.50, the cell's lowest coherent draw): "For a 7th birthday party, the decision between renting or buying a home depends on several factors…" — the target query arrives fused with the prefix topic.

Party-planning prefix, sky-color query edited toward the rent-vs-buy query (F = 0.52): "When deciding between renting or buying a home for your daughter's 7th birthday party, consider the following factors…"

Pirate prefix, sky-color query edited toward the hash-table query (F = 0.68): "Ahoy, me hearty! Aye, ye should use a hash function when ye need to quickly find information in a large collection of data…"

For contrast, a shuffled-donor null draw (pirate prefix, hash-table query, donor delta pointing at the same rent-vs-buy target, F = 1.02): "Ah, matey, that's a tricky question, like choosing between a plank or a cannonball. Rentin' a home is like havin' a ship without a crew…" — with only five queries in the bank, a wrong-pair donor often installs the same target query, which is what elevates the null.

</details>

For reference, a generate-under-target ceiling draw (pirate context, hash-table query): "Ahoy there, lippy! To explain 'bout hash tables, ye need to know first that they're like a big ol' treasure chest…" — all 150 anchor draws: [anchors @ pinned revision](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/cfedd60af9e061e9efa42a1573ebeab9ec790eca/issue2094_singlepos/raw_completions/anchors/anchors.jsonl).

## Results

### Behavior barely moves at the stage-1-selected cells once anchors separate

The heatmap maps mean behavior F per steered cell (rows = slots, columns = steered layer, dose facets, one panel per setting).

![Behavior fraction-of-swap heatmap across slots, layers, doses, and settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/hero2_f_beh_heatmap.png)

> **Figure.** *Nearly every single-layer cell reads F near zero.* Circles mark degenerate-by-design cells, x marks cells under 50% coherent; the saturated cells at joint variants and late context-end layers are the weak-anchor artifacts dissected in the next section.

On well-separated pairs the stage-1-selected best cells read 0.22 (cross, context-end layer 14, dose 4x; null 0.04; p = 0.16, n = 12) and 0.20 (matched query, context-end layer 14, full-state patch; null -0.02; p = 0.15, n = 10) — not separated from the shuffled-donor null at these cells, though the grid-wide screen in the final sections separates neighboring context-end cells cleanly.

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

Matched-query context-end edits move the answer state by a small, real amount: 0.16 steered vs 0.09 null at the banked layer 14, dose 1x, with overlapping bootstrap intervals (0.09 to 0.23 vs 0.02 to 0.17); the largest single-layer excess anywhere on the all-pairs grid is 0.14 (layer 14, dose 2x, prefix-centroid). The prefix-end slot moves more in raw terms but fails specificity outright: its best raw cell (layer 26, full-state patch) reads 0.210 steered vs 0.209 null. Under the well-separated restriction the activation read separates cleanly at nine context-end cells — the all-layer patch (0.41 vs null 0.09) and mid-stack joint edit at half dose (0.27 vs 0.03) lead — 5 of the 9 at cells the behavior screen also keeps, the other 4 at neighboring context-end layers and doses.

### The response to an injected edit is strongly non-linear, and banked maps do not transport

The figure shows the cosine between the realized answer-state shift and each banked map's predicted shift, per map and dose regime, steered vs shuffled-donor null, with per pair-dose points (tail-inclusive pooling for map parity).

![Banked-map transport cosines at context-end and prefix-end cells, steered vs null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/result1b_transport_cosines.png)

> **Figure.** *Banked maps predict almost none of the realized shift.* Bars are per-map means pooled over additive doses (left panel) and the full-state patch (right panel), steered vs null; points are pair-by-dose values. Pooled bars top 0.09; the best per-dose cell mean is 0.16 (context-end layer-14 map, dose 2x, prefix-centroid); the full-state context-map null matches its steered bar.

The realized shift's magnitude barely grows over an 8x dose range (mean log-log slope 0.00-0.06 by slot-layer family vs 1.0 for a linear map; the prefix-end families are fully dose-insensitive, slope 0.00 with median adjacent-dose cosine 1.00) and its direction drifts across doses at context-end (median adjacent-dose cosine 0.62 at layer 14; median split-half reliability 0.93, so the drift is not noise). The one-operator fit explains at most 8.4% of held-out response variance (2.3% family-held-out) and does not align with the banked map at layer 14 (Procrustes cosine 0.256 vs rotation-null 97.5th percentile 0.293). The additivity spot-check is weak where informative (cosine 0.17-0.88); its reported 1.000 comes from a combo whose three rollouts were the same greedy completion — insensitivity, not additivity.

### Single-position edits are safe; multi-position mid-stack edits break fluency non-specifically

The heatmaps show excess incoherence per cell (incoherent fraction minus the 2.0% unpatched anchor baseline) for steered and null arms, with the steered cap-hit fraction as a companion panel.

![Fragility heatmaps of excess incoherence for steered and null arms with cap-hit companion](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/result4_fragility.png)

> **Figure.** *Both arms stay coherent everywhere except query-span and last-three joint mid-stack cells.* Excess incoherence per slot x layer-variant x dose (n = 60-75 rollouts per cell arm), plus the steered cap-hit fraction.

Steered incoherence totals 0.7% (149 of 21,000) against the 2.0% unpatched baseline — single-position edits, full-state patches included, essentially never break fluency. The fragile cells are the multi-position controls at joint mid-stack layers: query-span at dose 4x reaches 80% incoherent (null 77%) and 73% cap-hit — generic disruption the wrong-pair null reproduces. Sixteen steered cells breached the plan's 2% cap-hit re-generation threshold: the query-span and last-three families (13-46% pooled by slot and setting over doses and arms; 5-73% per cell; last-three matched-prefix, 0.3%, did not breach) and the context-end joint-stack cells (2.7-6.7%) — the same cells that saturate the behavior heatmap — so their F values are untestable rather than transfer evidence, though several truncated query-span rollouts visibly moved query content while destroying fluency, the only sighting of query transfer in the grid.

### A grid-wide screen on well-separated pairs finds null-separated transfer only at context-end

The figure shows behavior F dose response at the two banked layers (14 and 19) on well-separated pairs: steered vs shuffled-donor null means with pair-clustered bootstrap bands (pair-difference edits), per-pair traces behind, one column per setting, context-end and prefix-end rows.

![Behavior F dose response at layers 14 and 19 on well-separated pairs, steered vs shuffled-donor null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/267387d2c9b448d3d0f29a1cb7bc8e43cd222e36/figures/issue_2094/result_fbeh_dose_L14L19_wellsep.png)

> **Figure.** *The steered band clears the null only in the matched-query column at context-end.* Dose response on well-separated pairs (n = 10-12 pairs per band; bands: pair-clustered bootstrap, B = 10,000); matched-prefix by prefix-end panels are degenerate by design.

Grid-wide, 36 of 1,245 behavior cell families read steered above null with fully disjoint 95% intervals, but 21 sit in the cap-hit-breached cells of the previous section, leaving 15 clean survivors — all context-end, 14 matched-query and 1 cross, all on the prefix rubric. Clean single layers 12-20 read 0.18-0.33 (nulls -0.24 to -0.00), the mid-stack joint edit at half dose 0.43, and the all-layer full-state patch 0.63 vs null -0.05 (n = 10 pairs).

Prefix-end and the single-token controls: zero. The screen is post hoc and unadjusted; each cell is one greedy draw per pair; the next section re-measures all 15 independently; the surviving cells carry zero intrusion flags.

### Executing the cap-hit re-generation trigger changes nothing, and 10 of the 15 surviving families confirm under independent re-sampling

The forest plot shows, per surviving family, the steered and shuffled-donor-null mean behavior F at temperature 1.0 with pair-clustered bootstrap 95% intervals and the per-pair means behind each interval; filled steered markers are the 10 families whose intervals are fully disjoint.

![Forest plot of steered vs shuffled-donor null behavior F for the 15 surviving families](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99448919d1972edd45bd12426dbc57c3fb4ff9e8/figures/issue_2094/fu1_conf1_forest.png)

> **Figure.** *Every steered mean sits above its shuffled-donor null; 10 of 15 families separate on fully disjoint intervals.* Steered and null means with pair-clustered bootstrap 95% intervals per family (B = 10,000; n = 10-12 well-separated pairs, 5 temperature-1.0 draws per pair), per-pair means behind; families ordered by steered mean.

Re-generating the 16 breached cells at a 2048-token cap leaves every one over the 2% trigger (residual steered cap-hit 2.7-72%): the long generations are genuine behavior of those cells, and substituting the re-generated rows changes nothing — 4,112 untouched families reproduce the parent read exactly, swap deltas stay within 0.023, and the clean set holds at 15. The confirmed families span single layers 13-17, both mid-stack joint variants, and the all-layer patch (0.51 vs null 0.10); 20 of 65 breached-cell families read separated at 2048 but stay excluded per the trigger. The confirmation is post-selection — a robustness read, not an unbiased estimate.

### Excluding template tokens makes the query span measurable; whole-prefix patches collapse coherence in both arms

The figure shows the 14 clean query-text family reads: steered and shuffled-donor null means with pair-clustered 95% intervals, per-pair reads behind, marker shape by read metric; the [verdict-class composition per span](https://raw.githubusercontent.com/superkaiba/explore-persona-space/30e2d438e92d735271af32bd44cd8d0deed06b1c/figures/issue_2094/fu2_verdict_composition.png) is linked, not embedded.

![The 14 clean query-text reads, steered vs null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/493ed29a255022bd0c5bec107bb3a71796837a70/figures/issue_2094/fu2_qtext_clean_forest.png)

> **Figure.** *Every clean query-text read separates from its null on disjoint 95% intervals.* Steered means run 0.17-0.95 of a full swap against nulls of -0.08 to 0.56 over all 14 reads, each at half dose or the mid-stack joint band; 10-29 usable well-separated pairs per read (8 circles query-rubric behavior, 5 squares activation, 1 triangle prefix-rubric).

Query-text edits yield 14 of 70 clean reads (27 counting cap-compromised); the 8 query-rubric behavior reads run 0.68-0.95 of a full swap against elevated nulls of 0.22-0.56. A shuffled donor still installs query text, often sharing the target from the 5-query bank, so the separation shows the injected delta directing which query the completion answers — content-directed transfer, not generic disruption. Pair-specificity beyond that stays unresolved: a matched-target wrong-pair null was not run.

Whole-prefix patches yield zero clean reads: 42 of 50 template-inclusive reads are not comparable (93% of steered and 97% of null completions incoherent), and content-only prefix patches retain majority coherence at half dose but never separate. These are one-greedy-draw screen reads, post hoc, not confirmation-grade; the four clean cells carry zero script-intrusion flags in either arm.

---

**Repro:** pod run at code SHA `cf7f2254ce` (grid + anchors, 1x H100, 42,000 rollouts, cap-hit 0.69%; stage-2 pod `pod-2094-s2`, 600 + 18 rollouts, cap-hit 0); off-pod analysis + figures at `5552c3e3c7`…`9d252d52c3` on branch `issue-2094`; follow-up analysis round (grid-wide well-separated bootstrap, banked-layer dose figure, corrected stage-1-vs-stage-2 join) at `267387d2c9`, code-review PASS, no new generation. GPU-backed follow-up round `fu1_regen_confirm` (proposer cheap band, ~2.5 GPU-h realized on `pod-2094-fu1`, 1x H100): driver `scripts/issue2094_fu1.py` at `c507e825a4`, judge outputs (18 waves) at `fd8eef2ecd`, swap + confirmation tables at `c0ae38c896` (`eval_results/issue_2094/f_metrics/fu1/fu1_regen_swap.json`, `fu1_wellsep_bootstrap_regen.json`, `fu1_conf1_confirmation.json`; judge work root `eval_results/issue_2094/judge_fu1/`), confirmation forest figure via `scripts/issue2094_fu1_fig.py` at `99448919d1`, code-review PASS x2 (driver + analysis). User-requested follow-up round `fu2_span_slots` (~2.4 GPU-h realized on `pod-2094-fu2`, 1x H100): driver `scripts/issue2094_fu2.py` at `76686b33b3` plus bank-integrity hot-fix `2087a6fcc8`, judge outputs (9 waves) at `b03264994b`, per-cell F reads + verdict table at `7d85517af5` (`eval_results/issue_2094/f_metrics/fu2/` — `fu2_cells.jsonl`, `fu2_null_cells.jsonl`, `fu2_f_reads.json`, `fu2_summary.json`; judge work root `eval_results/issue_2094/judge_fu2/`), fold figures via `scripts/issue2094_fu2_fig.py` at `30e2d438e9` (clean-read forest added to the same script at `493ed29a25` in the review round), code-review PASS x2 (driver + analysis); linked-not-embedded (deliberate): [coherence-by-dose per span, steered vs null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/30e2d438e92d735271af32bd44cd8d0deed06b1c/figures/issue_2094/fu2_coherence_by_dose.png), [verdict composition per span](https://raw.githubusercontent.com/superkaiba/explore-persona-space/30e2d438e92d735271af32bd44cd8d0deed06b1c/figures/issue_2094/fu2_verdict_composition.png). HF span-round artifacts @ 06ddddb7c1352d58917f4710cb0670bfc6024a22 (verified live via `list_repo_tree`): `issue2094_singlepos/raw_completions/fu2_span_slots/rollouts` ([pinned tree](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/06ddddb7c1352d58917f4710cb0670bfc6024a22/issue2094_singlepos/raw_completions/fu2_span_slots/rollouts), 60 shards), `issue2094_singlepos/analysis_tensors/fu2_span_slots/` (`va`, `bank` incl. the fresh state bank + capture-parity report, `manifests`), `issue2094_singlepos/raw_completions/judge_raw_fu2`. Figures: `figures/issue_2094/` (18 run figures + `result_sep_stratified_fbeh` via `scripts/issue2094_sep_stratified_fig.py` + the analysis round's two + `fu1_conf1_forest`); linked-not-embedded (deliberate): [per-pair anchor separations](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/exp_anchor_separation.png), [activation dose curves](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d252d52c3be23f43843cc97ac9913d0adace67f/figures/issue_2094/hero1_f_act_dose_curves.png), [stage-1 vs stage-2 selected-cell scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/267387d2c9b448d3d0f29a1cb7bc8e43cd222e36/figures/issue_2094/exp_stage2_vs_stage1.png) (re-rendered in the follow-up round; the run's original render mis-joined the stage-1 cells and drew a fallback axis). Eval tables: `eval_results/issue_2094/` (`f_metrics/f_cells.jsonl`, `null_cells.jsonl`, `anchors.jsonl`, `bootstrap_cis.json`, `bootstrap_cis_wellsep.json`; `linearity/`, `transport/`, `fragility/`, `judge/`, `gates/`, `best_cells.json`, `map_parity.json`; `f_metrics/fu1/`, `judge_fu1/`). HF (verified live via `list_repo_tree`): [issue2094_singlepos](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/cfedd60af9e061e9efa42a1573ebeab9ec790eca/issue2094_singlepos) — `raw_completions/{grid,anchors,stage2,stage2_additivity,judge_raw}` @ cfedd60af9e061e9efa42a1573ebeab9ec790eca (880 grid shards) plus analysis tensors (`va_store`, `vc_bank`, stage-2 stores); follow-up round artifacts @ ce5a712423fb9bd8c3c4bda91c4886d28865d643 — `issue2094_singlepos/raw_completions/fu1_regen_confirm/regen` (44 shards), `issue2094_singlepos/raw_completions/fu1_regen_confirm/conf1` (30 files), `issue2094_singlepos/analysis_tensors/fu1_regen_confirm/` (74 tensors + manifests), `issue2094_singlepos/raw_completions/judge_raw_fu1`. Reused artifacts: ridge maps from [#779](https://eps.superkaiba.com/tasks/779) (`issue779_monitoring/n1m_readout/weights/{L14,L19,L26}/ridge.pt` @ cfedd60af9e061e9efa42a1573ebeab9ec790eca) and [#1738](https://eps.superkaiba.com/tasks/1738) (`issue1738_multiturn/analysis_tensors/weights/{L14,L19,L26}/prefix_ridge.pt` @ cfedd60af9e061e9efa42a1573ebeab9ec790eca) — fit: transport-only frozen instruments, input-convention parity recorded in `map_parity.json`; Jacobian `J_last.pt` from [#1776](https://eps.superkaiba.com/tasks/1776). All behavioral reads are on-policy generations (greedy grid; temperature-1.0 anchors, stage-2, and confirmation draws; greedy re-generation); no teacher forcing. Known deviations (all recorded in the run digest): per-position add realization of multi-position replace doses; retokenized cap-hit basis; the null installs the donor's target state (not a difference) at replace cells; matched-prefix prefix-end degeneracy.

**Context:** originating prompt (verbatim): "start in background with happy coder and setup periodic monitoring of it (design finalized in chat 2026-08-05: unified F metric, crossed 3-prefix x 5-query bank, 3 settings, slot grid incl. prefix-end injection, greedy 1-draw grid + K=10 anchors, coherence>60 gating, fragility map, banked-map-only transport, stage-2 best-cell confirmation at temp 1.0 K=5)". Design finalized interactively in chat 2026-08-05; run 2026-08-06 (grid pod ~4 h; judge waves + stage-2 same day); free-analysis follow-up round folded 2026-08-06. GPU-backed follow-up round `fu1_regen_confirm` (proposer cheap band, auto-run) executed and folded 2026-08-06: the plan's cap-hit re-generation trigger executed at a 2048 cap plus independent re-measurement of the 15 surviving families. Third same-issue follow-up round `fu2_span_slots` (source: user-chat) run 2026-08-07 and folded the same day; originating prompt (verbatim): "can you run: - query only (without template tokens) - prefix only (both with and without template tokens) all layers + middle band". Lineage: steering conventions from [#1415](https://eps.superkaiba.com/tasks/1415); banked maps [#779](https://eps.superkaiba.com/tasks/779)/[#1738](https://eps.superkaiba.com/tasks/1738); Jacobian [#1776](https://eps.superkaiba.com/tasks/1776).

