# Methodology — issue 2094

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

**Evaluation:** behavior F per draw = (judge contrast of the rollout minus the floor mean) / (ceiling mean minus floor mean); judge contrast = (score against descriptor B minus score against descriptor A)/100; matched-prefix pairs use query rubrics, matched-query pairs prefix rubrics, cross pairs both. Activation F = signed projection of the patched-minus-floor answer-state shift onto the ceiling-minus-floor axis, floor split into disjoint halves. All reported quantities use coherent draws only. "Well-separated" below means absolute anchor separation of at least 0.5 (judge-contrast units; maximum 2). Anchor separations on the prefix rubric: 5 of 15 matched-query and 3 of 15 cross pairs fall below 0.5, of which 3 and 2 sit at 0.03 or less, and one retained cross pair is inverted at -0.96 (its ceiling judged below its floor); all matched-prefix and cross query-rubric separations are at least 0.5. The plan specified per-pair separation reporting with no exclusion rule; the well-separated stratification in Results was chosen after seeing the data, not written into the plan. The grid-wide screen recounts the pair-clustered bootstrap under that restriction with the parent run's own batched implementation, exclusions, and gating: per cell family (setting x slot x layer variant x dose x edit-direction type x metric), steered and null means with 95% intervals over well-separated pairs — 2,172 steered-vs-null comparisons at five or more pairs (927 activation, 757 prefix-rubric, 488 query-rubric behavior). "Separates" in Results means fully disjoint 95% intervals with steered above (4 further behavior families are disjoint with steered below, all four at prefix-end); the weaker read — null mean outside the steered interval — fires on 223 prefix-rubric, 38 query-rubric, and 117 activation families in either direction and carries no claims. The screen is post hoc, with no multiplicity adjustment across families. 148,650 judge calls in 27 Batch-API waves: 25 content drops, 0 residual transport losses, 0 truncation drops; the judge pilot gate, the anchor coherence gate (median 100, 98% of draws above 60), and the injection-exactness gate (12 of 12 spot cells) all passed before production. Paired steered-vs-null p-values in Results are exact signed-rank tests over pairs; steered-vs-null interval comparisons use the pair-clustered bootstrap. Cell-level agreement between the two F levels: Spearman rho = 0.43-0.50 across 929 steered cell families (p < 1e-40), so the continuous activation companion tracks the judged behavioral read. Mechanical audits (script intrusion at non-Latin letter fraction > 0.05, repetition, empty output) ran on every arm: grid 214 of 21,000 steered vs 239 of 21,000 null flagged for script intrusion (repetition: 111 vs 96; none empty); anchors 7 of 150; stage-2 31 of 600 (both intrusion-only); intrusion-excluded recounts of every stage-2 cell shift coherence-gated means by at most 0.02, and the 15 surviving screen cells carry zero intrusion flags in either arm (225 rows each).

**Data extraction:** per-context state vectors captured in one right-padded forward pass per context (positions read off token ids, never re-tokenized strings); answer vectors are span-means over the model's own completion at all 28 layers, in both the parent convention and the tail-inclusive pooling used for map transport (parity of each banked map's input convention with the injected slot recorded in `eval_results/issue_2094/map_parity.json` before any transport number). Every F table row carries pair, slot, layer variant, dose, coherence flag, cap-hit flag, and donor pair id.

Acknowledged conciseness overages (deliberate, WARN-tier): several Takeaways bullets exceed 30 words and six Results blocks exceed 120 words — the denominator-artifact dissection, the non-linearity battery, the breach disclosures, and the grid-wide screen's reconciliation each need the extra clauses to stay verifiable against the artifacts; total content prose runs over the 800-word budget for the same reason.

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
