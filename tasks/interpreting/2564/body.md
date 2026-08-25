---
goal: 'On Qwen-2.5-7B-Instruct, using a maximally-controlled single-turn minimal-pair
  battery (one token-matched instruction or query manipulation per pair, no crossing)
  across instruction axes (persona, format, lexical-marker, stance, content-constraint,
  register, hedging, single-token-user-fact-with-express-instruction, diffuse-user-profile-with-be-aware-instruction)
  and query axes (content, form), measure how faithfully the frozen #779 context-end
  ridge map predicts the DIFFERENCE between paired answer representations — per axis:
  direction, magnitude calibration vs global shrinkage, real-vs-predicted axis-identity
  cosine, and split-half reliability — against an identity+bias delta baseline, a
  paraphrase null, and a Qwen3-Embedding-8B answer-text third space, to determine
  which kinds of context information the map carries faithfully vs distorts.'
title: The frozen context-to-answer map recovers the direction of instruction-driven
  answer shifts on every minimal-pair axis that passed its compliance check, beating
  identity pass-through throughout (MODERATE confidence)
has_clean_result: true
---
# The frozen context-to-answer map recovers the direction of instruction-driven answer shifts on every minimal-pair axis that passed its compliance check, beating identity pass-through throughout (MODERATE confidence)
<!-- clean-result-v4 -->

## Takeaways

- On the 6 of 9 instruction axes that passed the on-policy compliance gate (7 of 9 pass at a 50% comply threshold, 5 of 9 at 90%), plus both query axes, the frozen single-turn map's predicted answer-shift direction beats chance and the identity baseline everywhere: per-axis mean cosine between predicted and observed answer-state differences runs 0.17 to 0.90 against shuffled-pair nulls whose 97.5th percentiles sit at 0.04 to 0.45, and the paired map-minus-identity gap is +0.03 to +0.41 with every bootstrap interval excluding zero (12 to 120 pairs per axis).
- The tone instruction (very formal vs very casual) is carried almost perfectly: direction cosine 0.90 against a split-half reliability of 0.99, within-axis consistency 0.93 vs 0.76 for identity pass-through, and a magnitude slope at the map's global calibration (ratio 1.03, interval spanning 1).
- The injected-name read is reliability-limited rather than diagnosably map-limited: the observed shift's split-half reliability is 0.13 (every other tested axis is at or above 0.60), which caps a perfect predictor near cosine 0.36; the observed 0.17 recovers about 48% of that bound, in line with the other axes, and a reliability this low means further map-direction loss can be neither shown nor excluded, even though the name appears in 589 of 600 base-value rollouts.
- 3 of 9 instruction axes failed the compliance floor and are not tested: hedging (1 of 2 values fired), persona (2 of 5), stance (1 of 5). Qwen-2.5-7B-Instruct would not reliably argue one-sided stances, play most personas, or answer in strong-confidence mode on these debatable advice questions, so cross-axis claims use the 6-axis denominator and those rows appear as null bands only, never zero bars.
- The predicted difference finds its true pair at 23.7% top-1 among 2,778 candidates (chance 0.04%) under cosine, ahead of identity at top-1 and top-5 (identity edges ahead at top-10), but under euclidean distance identity wins at every depth (17.1% vs 15.3% top-1): the map's retrieval margin is metric-dependent, so the direction claims rest on the paired cosine gaps. The single-turn and multi-turn frozen maps agree on every tested axis (largest direction-cosine difference 0.047).
- 357 of 9,840 rollouts (3.6%) contain CJK-script text (worst axis: persona at 8.4%). Every compliance verdict survives both intrusion recounts except the user-description axis, whose pass is convention-dependent (2 of 5 values fired under zeroing vs 3 of 5 as scored and under exclusion); its direction read on the two intrusion-robust values keeps the same ordering (map 0.49 vs identity 0.36).

## Goal

Decompose, by kind of context information, what the frozen context-to-answer map actually carries: build a token-matched minimal-pair battery over 9 instruction axes and 2 query axes on Qwen-2.5-7B-Instruct, verify per value that each instruction changed behavior on policy, and test whether the map's predicted answer-state difference matches the observed one in direction and magnitude, against an identity pass-through baseline and shuffled-pair nulls.

**This experiment in context:** this is the designed follow-through of [#2215](https://eps.superkaiba.com/tasks/2215), which found that banked context-to-answer maps discriminate minimal-pair contexts but that an identity-plus-bias baseline captures most of that discrimination at context-end. #2215 scored retrieval-style discrimination over a smaller pair set under a different read protocol, so its numbers are not directly comparable; this battery instead scores the predicted-vs-observed answer-shift *difference* per axis, where any constant bias cancels exactly and the map-vs-identity question becomes a paired directional test. The single-turn map under test comes from [#779](https://eps.superkaiba.com/tasks/779), its multi-turn twin from [#1738](https://eps.superkaiba.com/tasks/1738), and the rollout/capture rig from [#2162](https://eps.superkaiba.com/tasks/2162).

**Broader narrative:** the context-to-answer line of the project asks whether a single linear map from context-end hidden states predicts the answer representation a context will produce. Aggregate held-out fit says yes on pooled corpora; this battery says *which kinds* of context information survive the map (tone, query topic, audience description, content constraints) and which reads are limited by sampling noise or by the model's own unwillingness to comply, anchoring every representational claim to a behavioral manipulation check.

Conciseness acknowledgment: this single-round body deliberately ships three check-20 WARN classes (Takeaways bullet length over the 30-word guidance, per-result prose over the 120-word band on all seven results, and total prose over the 800-word budget) because the battery reports 11 axes with per-axis nulls, reliabilities, baselines, and intrusion recounts that lose meaning when split.

## Methodology

**Design:** 984 single-turn contexts: 12 everyday advice questions (carriers) crossed with 39 instruction values spanning 9 axes (tone, output format, hedging level, marker-word insertion, assumed persona, stance, content constraint, injected user name, user description), each with one wording paraphrase, plus an empty-system anchor and query-content/query-form variants. From these, 2,778 pairs in 7 classes: value swap (864), install vs empty system (468), instruction-wording paraphrase (468), family swap across wording families (864), query content (66), query form (36), query paraphrase (12). Each context gets K=10 sampled responses; the observed shift for a pair is the difference of the two contexts' 10-draw-mean answer representations at layer 19. Three predictors score identical rows: the frozen single-turn ridge map under test (primary), its frozen multi-turn twin, and identity pass-through (the map's learned bias cancels exactly in differences; verified numerically to 3.6e-15 over 32 pairs). Headline per-axis reads use only values that passed the compliance gate; gate-failing axes are reported not tested.

**Training:** **N/A — no model training.**

| Parameter | Value | Source |
|---|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, bf16, frozen | plan §3, inherited from #2162/#2215 |
| Rollouts | K=10 per context, temperature 1.0, seed 42, `max_new_tokens` 2048 (realized cap-hit 0.0% on all cells) | plan §0/§7; run sentinel |
| Capture layer | 19 primary; twin reads at 14 and 26 | #779 fit convention (plan §3) |
| Answer summary | tail-inclusive answer mean; span-mean pooling twin | #2215 parity-gated convention (plan §3) |
| Context vector | context-end last-token hidden state | #779 fit convention |
| Scored arms | frozen #779 single-turn ridge; frozen #1738 multi-turn ridge; identity pass-through | plan §5 |
| Bootstrap | carrier-clustered, B=10,000, seed 2215 | plan §6 |
| Null | per-axis value-pair derangement, B=10,000, seed 21620; 2-value axes use a named sign-randomization null | plan §6 |
| Split-half reliability | 20 seeded 5-vs-5 rollout splits, stepped up to the 10-draw mean | plan §6 |
| Judge | `claude-sonnet-4-5-20250929`, Batch API, `max_tokens` 1024, temperature at the API default | plan §6; project judge pin |
| Fire rule | a value fires at >= 70% of its 24 checks complying (judge score >= 50); any incomplete check makes the value undetermined (counted not fired); axis floor = ceil(0.6 x width); 50% and 90% comply-threshold sensitivity columns reported | plan §6 |
| Text space | `Qwen3-Embedding-8B` mean answer embeddings (third-space control) | plan §6 |
| Generation / capture batch | 16 / 8 | plan §9; run marker |

**Evaluation:** the primary DV per pair is the cosine between the predicted and observed answer-state differences, averaged per axis over swap pairs of fired values, with a carrier-clustered bootstrap interval, a per-axis shuffled-pair null, and a paired map-minus-identity gap under the shared bootstrap. Companions per axis: magnitude calibration (regression slope of predicted on observed shift norms as a ratio to the map's global slope 1.014 over all 2,778 pairs, with a per-value-pair decomposition committed for the content-constraint axis), within-axis consistency (median cosine among the axis's per-carrier observed directions as reproduced by each arm; undefined for the query-content axis, whose pairs are carrier dyads with no carrier-replicated value pair), cross-wording-family consistency, split-half reliability of the observed shift, and a text-embedding third space. Global retrieval asks whether each predicted difference finds its own pair among all 2,778 under cosine and euclidean distance (identity baseline and kNN reads reported per the project's representation-mapping rule). The manipulation check judges 2 draws per (value, carrier) for the 7 judged axes (1,392 draws, completeness 0.981, 16 API refusal draws + 11 dropped) and checks all 10 draws programmatically for the marker-word and injected-name axes (word/name containment, denominator 120). A language-intrusion audit scans every rollout and the judged pool for CJK script and recomputes all fire verdicts with intruded draws zeroed and excluded; its per-arm counts, judged-pool cross-tab, and full recount table are committed alongside the eval JSONs.

**Data extraction:** bank build and gate battery (7 gates, all PASS) on the VM; generation plus teacher-forced capture pod-side (capture wall 3,352 s); per-cell answer summaries, context states, text embeddings, and predictions uploaded to the HF data repo before pod release; judging via the Batch API and all statistics off-pod from the committed artifacts. The identity-cancellation assert and the fixed-answer equivalence gates ran in production, not only under smoke.

**Sample training/evaluation data + completions:**

<details>
<summary>Tone instruction (headline axis): 3 complying rollouts; all 48 judged tone checks complied, so no non-complying rows exist for this axis</summary>

Random sample (seed 42): 3 of 48 complying tone base-value checks; all rows: [rollout shards on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/62b1e8889e1a262501937b0ec6f6022e28b4a7e6/issue2564_minpair/raw_completions/anchors).

1. **System:** Use a very casual and conversational tone throughout. **User:** How should a person choose a career? **Completion (comply, score 82; truncated):** "Hey there! Picking a career is like choosing the perfect pair of sneakers—the fit matters a ton! Here's how you can make a decision that feels right for you: 1. **Figure Out What You Love**: Think about what you genuinely enjoy doing..."
2. **System:** Use a very formal and highly professional tone throughout. **User:** Should students be required to learn coding? **Completion (comply, score 100; truncated):** "The integration of coding education into the curriculum of academic institutions is a topic of considerable importance and merit in contemporary educational discourse..."
3. **System:** Use a very formal and highly professional tone throughout. **User:** Should I adopt a dog or a cat? **Completion (comply, score 100; truncated):** "When considering the adoption of a pet, whether to opt for a canine or feline companion, it is crucial to evaluate several variables that resonate with your lifestyle..."

</details>

<details>
<summary>Stance instruction (compliance-floor FAIL): 3 non-complying and 3 complying checks</summary>

Random sample (seed 42): 3 of 69 non-complying and 3 of 51 complying stance base-value checks; all rows: [judge outputs on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/62b1e8889e1a262501937b0ec6f6022e28b4a7e6/issue2564_minpair/raw_completions/judge).

1. **System:** Argue against every option in your answer. **User:** Is it better to exercise in the morning or evening? **Completion (noncomply, score 40; truncated):** "Let's evaluate both options: Morning Exercise: 1. Morning exercise can boost your metabolism for the entire day..." (lists benefits before any counterargument).
2. **System:** Argue against every option in your answer. **User:** Is it worth traveling somewhere alone? **Completion (noncomply, score 0; truncated):** "Traveling alone may not be the best option for everyone, but let's consider why it might not be worthwhile: 1. **Safety Concerns**..." (argues against one option only).
3. **System:** Argue against every option in your answer. **User:** Is remote work better than working in an office? **Completion (noncomply, score 35; truncated):** "Remote work is not universally better than working in an office for everyone or in all situations..."
4. **System:** Argue against every option in your answer. **User:** How should a person choose a career? **Completion (comply, score 70; truncated):** "Choosing a career is a complex decision with many important factors to consider, and here are arguments against some common approaches people might take: 1. **"Just Follow Your Passion":** This can be limiting..."
5. **System:** Play devil's advocate constantly in your answer. **User:** What's the best way to make new friends? **Completion (comply, score 65; truncated):** "Making new friends can be approached from various angles, and while many methods are effective, let's explore a Devil's Advocate perspective..."
6. **System:** Argue against every option in your answer. **User:** Should I read more fiction or nonfiction? **Completion (comply, score 65; truncated):** "Reading more fiction can have its drawbacks, as it might limit your exposure to real-world issues..."

</details>

<details>
<summary>Injected-name instruction (programmatic check): 3 firing and 3 non-firing rollouts</summary>

Random sample (seed 42): 3 of 589 firing and 3 of 11 non-firing injected-name base-value rollouts (600 total); all rows: [rollout shards on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/62b1e8889e1a262501937b0ec6f6022e28b4a7e6/issue2564_minpair/raw_completions/anchors).

1. **System:** The user's name is Diego. Always address them by name. **User:** Is it better to exercise in the morning or evening? **Completion (firing; truncated):** "That depends on your personal schedule and preferences, Diego. Many people find that exercising in the morning can help boost their energy levels..."
2. **System:** The user's name is Marcus. Always address them by name. **User:** How should someone spend a free weekend? **Completion (firing; truncated):** "Hello Marcus! A free weekend can be a great opportunity to recharge and do some activities you enjoy..."
3. **System:** The user's name is Sarah. Always address them by name. **User:** How should a person choose a career? **Completion (firing; truncated):** "Choosing a career is a big decision, Sarah! Here are some steps you might find helpful..."
4. **System:** The user's name is Diego. Always address them by name. **User:** Should students be required to learn coding? **Completion (non-firing; truncated):** "Absolutely, I think students should be required to learn coding. In today's digital age, basic programming skills are becoming increasingly important..." (never uses the name).
5. **System:** The user's name is Sarah. Always address them by name. **User:** How should a person choose a career? **Completion (non-firing; truncated):** "Choosing a career is an important decision that can have a significant impact on your life, so it's great to approach it thoughtfully!..." (never uses the name).
6. **System:** The user's name is Marcus. Always address them by name. **User:** Should I read more fiction or nonfiction? **Completion (non-firing; truncated):** "That depends on what you enjoy and what you hope to gain from your reading!..." (never uses the name).

</details>

## Results

### Wherever the instruction fired, the map recovers the answer-shift direction and beats identity pass-through

Plotted: for each axis, the mean cosine between the predicted and observed answer-state differences over token-matched value-swap pairs of fired values (panel 1, with the shuffled-pair null band in grey and the split-half reliability as a diamond), the magnitude slope ratio to the map's global calibration (panel 2), within-axis consistency (panel 3), and the text-vs-representation flip-to-paraphrase ratio (panel 4); blue = single-turn map, orange = multi-turn twin, green = identity baseline.

![Per-axis profile of the frozen maps: direction recovery, calibration, within-axis consistency, and text-vs-representation panels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d313b856460ea364de2aa12343e380eb37a73a62/figures/issue_2564/fig_hero_axis_profile.png)

> **Figure.** *Direction recovery clears the null on all 8 tested axes with the map above identity on every one.* Whiskers are carrier-clustered bootstrap intervals; grey bands are shuffled-pair null 97.5% ranges; diamonds mark the observed shift's split-half reliability; hedging, persona, and stance show only null bands (floor failed); the query-content consistency cell is blank (undefined for carrier-dyad pairs).

| Axis | Pairs | Map cosine [95%] | Null 97.5% | Identity cosine | Map minus identity [95%] |
|---|---|---|---|---|---|
| tone | 12 | 0.899 [0.894, 0.904] | 0.452 | 0.728 | +0.171 [0.163, 0.178] |
| query content | 66 | 0.715 [0.675, 0.751] | 0.096 | 0.304 | +0.410 [0.387, 0.433] |
| user description | 120 | 0.596 [0.568, 0.627] | 0.047 | 0.427 | +0.169 [0.144, 0.197] |
| content constraint | 120 | 0.497 [0.479, 0.516] | 0.155 | 0.314 | +0.184 [0.169, 0.199] |
| marker word | 120 | 0.416 [0.401, 0.429] | 0.071 | 0.273 | +0.143 [0.124, 0.159] |
| output format | 120 | 0.362 [0.340, 0.379] | 0.042 | 0.229 | +0.133 [0.110, 0.152] |
| query form | 36 | 0.304 [0.269, 0.335] | 0.077 | 0.221 | +0.083 [0.058, 0.114] |
| injected name | 120 | 0.172 [0.152, 0.193] | 0.036 | 0.138 | +0.034 [0.018, 0.049] |

Every tested axis clears its null and every paired gap excludes zero, so the map carries per-kind information beyond raw context-state geometry. The two frozen maps agree throughout (largest difference 0.047, on query content), and cross-wording-family medians of 0.72 to 0.97 show the recovered directions are not template artifacts. One caveat on tone: the manipulation also lengthens answers (mean +108 tokens, formal minus casual), though the span-mean pooling twin reproduces the read (0.898), so pooling choice is not driving it.

### Per-pair spread behind the axis means: paraphrase pairs read lowest, swap and install pairs agree

Plotted: the full per-pair direction-cosine distributions for the single-turn map, grouped by axis with one violin per pair class (value swap, family swap, install vs empty system, instruction-wording paraphrase; each query axis carries its own class), fire mask not applied, so the three floor-failing axes appear here as raw views.

![Per-pair direction cosine distributions by axis and pair class for the single-turn frozen map](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d313b856460ea364de2aa12343e380eb37a73a62/figures/issue_2564/fig_expl_install_vs_swap_violin.png)

> **Figure.** *The per-pair clouds behind the headline means.* Violins show 2,766 of the 2,778 pairs (the 12 query-paraphrase pairs appear only in the query-axis figure), medians and quartiles marked; value-swap and family-swap distributions largely overlap on every axis, wording-paraphrase pairs sit lowest on the content-bearing axes, and the not-tested axes are shown raw.

Value swaps and family swaps track each other closely on every axis (largest mean offsets: stance 0.54 vs 0.48, injected name 0.17 vs 0.23), and instruction-wording paraphrases sit well below matched swaps on content-bearing axes (output format 0.19 vs 0.59 on the same all-values basis; tone 0.42 vs 0.90), confirming the pairs measure the manipulated variable rather than surface wording. Install pairs (instruction vs empty system) sit near swaps on most axes and above them for the injected name (0.39 vs 0.17): adding a name to an empty system prompt is a larger, more predictable move than exchanging one name for another, consistent with the reliability-limited swap read below.

### Six of nine instruction axes passed the compliance gate; stance, persona, and hedging did not

Plotted: per-base-value compliance fraction from the manipulation check (judge-scored for 7 axes, programmatic containment for the marker-word and injected-name axes), one bar per value with its plain-English instruction, colored by verdict, with the 70% fire threshold dashed.

![Per-value compliance fractions against the 70 percent fire threshold, colored by fired, not fired, or undetermined verdict](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d313b856460ea364de2aa12343e380eb37a73a62/figures/issue_2564/fig_expl_manip_fire_rates.png)

> **Figure.** *Where the behavioral gate passed and failed.* Green bars fired (at least 70% of checks complied), red did not, orange are undetermined (incomplete judge checks count as not fired). Stance, persona, and hedging miss their axis floors; the six other axes pass.

The gate is conservative and did real work: the model complied near-ceiling with tone, marker-word, name, format, and content-constraint instructions, but would not argue one-sided stances (back one option 2/24, devil's advocate 5/24), played 2 of 5 personas, and kept hedging (strong confidence 11/24). The denominator is threshold-sensitive: 7 of 9 instruction axes pass at a 50% comply threshold (assumed persona joins), 5 of 9 at 90% (user description drops). Zeroing CJK-intruded draws flips only the user-description axis (the college-freshman value falls 18/24 to 16/24, floor 2 of 5); excluding them restores 3 of 5, so that pass is convention-dependent; every other verdict, including the programmatic axes at worst-case zeroing (minimum 0.950), is intrusion-robust.

### Direction recovery tracks the reliability of the observed shift; the injected-name read is reliability-limited

Plotted: each tested axis's direction cosine (single-turn map) against the split-half reliability of its observed 10-draw-mean shift, one labeled point per axis.

![Direction cosine per axis plotted against the split-half reliability of the observed shift](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d313b856460ea364de2aa12343e380eb37a73a62/figures/issue_2564/fig_expl_ceiling_vs_cos.png)

> **Figure.** *Low raw recovery co-occurs with low split-half reliability.* Tone and query content pair reliabilities of 0.99 and 0.98 with high recovery; the injected-name axis sits at reliability 0.13 with cosine 0.17, where the bound a perfect predictor could reach is about 0.36.

Recovery orders with reliability: axes whose observed shifts replicate across rollout halves are the axes the map predicts well. A noisy target caps the cosine a perfect predictor can reach near the square root of the target's reliability, so the injected-name bound is about 0.36; the observed 0.17 recovers about 48% of it, essentially the same fraction as content constraint (51%), marker word (48%), and format (38%).

The low raw cosine is reliability-limited: the shift barely rises above sampling noise even though the name is expressed in 589 of 600 base-value rollouts. A reliability this low also means further map-direction loss on this axis can be neither demonstrated nor ruled out. Query form (reliability 0.60, cosine 0.30) keeps the clearest headroom.

### Magnitude calibration is heterogeneous and largely inherited: two content axes compressed, query form and the fired format subset over-scaled

Plotted: per-pair predicted vs observed shift norms for the single-turn map, one panel per axis, filled points for headline (fired) pairs and open points otherwise, with the map's global slope line (1.014) in every panel.

![Per-pair predicted versus observed answer-shift magnitudes by axis with the global calibration line](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d313b856460ea364de2aa12343e380eb37a73a62/figures/issue_2564/fig_expl_norm_scatter.png)

> **Figure.** *One global gain does not fit all axes.* Tone and marker-word clouds ride the global line; content constraint and query content fall below it (compressed); query form sits above it (over-scaled). The hedging cloud also sits above the line, an informational all-values view only: that axis failed its compliance floor. Filled points are fired-value pairs.

Two axes are compressed with intervals excluding 1: content constraint at 0.84 and query content at 0.80. User description (0.93) and injected name (0.87) sit below 1 with intervals spanning it; tone and marker word (both 1.03) ride the global gain; query form is over-scaled at 1.43 and the fired format subset at 1.19 (all format values: 1.02).

The distortion is not map-specific: identity pass-through is as compressed on the two content axes (0.85, 0.74) and more over-scaled on query form (2.12) and fired format (1.72), so the map partly corrects an inherited pattern. Within content constraint the compression is value-specific: the four pairs involving the under-twenty-words value, the largest and most length-coupled shifts, sit at 0.77 to 0.83, and excluding them the fired-pair slope is 0.99.

### The pair is identifiable among 2,778 candidates, but the map's margin over identity is metric-dependent

Plotted: retrieval accuracy at k = 1, 5, 10 for finding each pair's observed difference from its predicted difference among all 2,778 candidates, under cosine (left) and euclidean (right) distance, for both frozen maps and the identity baseline, with chance dashed.

![Retrieval accuracy at one, five, and ten candidates under cosine and euclidean distance for both maps and the identity baseline](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d313b856460ea364de2aa12343e380eb37a73a62/figures/issue_2564/fig_expl_retrieval_acc.png)

> **Figure.** *Strong retrieval, metric-dependent ranking.* Under cosine the single-turn map leads at top-1 and top-5 (23.7% top-1, median rank 6, chance 0.04%) while identity edges ahead at top-10; under euclidean the identity baseline leads at every depth.

The maps and the identity baseline all identify pairs far above chance, so answer-shift geometry is pair-specific. But the ordering flips with the metric: cosine favors the map at top-1 and top-5 (23.7% vs 21.8% top-1) with identity ahead by top-10 (63.4% vs 60.8%), and euclidean favors identity at every depth (17.1% vs 15.3% top-1), because the map's magnitude miscalibrations (previous section) cost it under a norm-sensitive distance while direction-only comparison rewards it. Retrieval alone cannot arbitrate map vs identity here; the paired per-axis direction gaps, which control the pairing and the metric, are the arbitrating read, and they favor the map on all 8 tested axes.

### A query wording change moves the answer state at paraphrase scale while changing the answer text less than a paraphrase

Plotted: the ratio of flip-induced change to paraphrase-induced change for the two query axes, measured in three spaces: answer text embeddings, observed answer states, and map-predicted answer states.

![Flip to paraphrase change ratios for query content and query form in text space, observed state space, and predicted state space](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d313b856460ea364de2aa12343e380eb37a73a62/figures/issue_2564/fig_expl_query_dissociation.png)

> **Figure.** *Query form dissociates text from state.* A query-content flip exceeds the paraphrase yardstick in every space (text 2.9, observed 4.2, predicted 2.6); a query-form flip changes answer text less than a paraphrase (0.66) while the state moves at paraphrase scale (observed 0.97, predicted 1.13).

Changing what the question asks moves everything, as expected. Changing only how it is phrased produces answers that read like paraphrases in text space, yet the answer state shifts as much as a real paraphrase, in a direction the map partially predicts (cosine 0.30 vs null 0.08).

The map therefore carries some purely formal query information that barely surfaces in the emitted text. The read is preliminary: 36 pairs, a 0.60 reliability, and the battery's largest over-scaling (1.43).

---

**Repro:** 2.2 GPU-h realized of 6 budgeted (generation + teacher-forced capture on 1x H100, RunPod `pod-2564`, backend runpod; judging via the Anthropic Batch API, 1,392 calls; bank build and statistics off-pod). Code at commit `09750cbbab8918d6767bdd8f8d65028355811b93` on branch `issue-2564` (`scripts/issue2564_run.py`, `scripts/issue2564_judge.py`, `scripts/issue2564_analysis.py`, figures via `scripts/issue2564_figures.py`); round-2 revision artifacts at commit `d313b856460ea364de2aa12343e380eb37a73a62` (intrusion audit via `scripts/issue2564_intrusion_audit.py`, content-constraint pair-slope decomposition via `scripts/issue2564_pair_slopes.py`); figures pinned at `d313b856460ea364de2aa12343e380eb37a73a62`. Eval artifacts: `eval_results/issue_2564/` (`minpair_delta.json`, `manipulation_check.json`, `perpair.jsonl`, `bank_manifest.json`, `intrusion_audit.json`, `content_constraint_pair_slopes.json`, `predictions/`); instruction-values file sha256 `29e6bebc52b91af5e92474d829ab7ee252f6c8b5f067696f3444d41f2454119d`. HF data: [issue2564 minpair tree](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/62b1e8889e1a262501937b0ec6f6022e28b4a7e6/issue2564_minpair) (raw completions: 10 rollout shards + judge outputs; analysis tensors: per-cell answer summaries `va2564`, context states `vc2564`, text embeddings, predictions; manifests). Model `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`; seeds: rollouts 42, bootstrap 2215, nulls 21620. Judge `claude-sonnet-4-5-20250929`.
- Reused map from [#779](https://eps.superkaiba.com/tasks/779): `issue779_monitoring/n1m_readout/weights/L19/ridge.pt` @ 62b1e8889e1a262501937b0ec6f6022e28b4a7e6 (sha256 `188486f8afd9d95221e32492f3a0be2a3bdb2098cbe7fadfecf1d46433567909`) — fit: the frozen single-turn context-to-answer ridge at layer 19, the exact production map under test; listing verified live at write time.
- Reused map from [#1738](https://eps.superkaiba.com/tasks/1738): `issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt` @ 62b1e8889e1a262501937b0ec6f6022e28b4a7e6 (sha256 `1916bed192f36dfe9dc57a3c712d3ea56d3767199504693c0c0f33bcde27bda4`) — fit: the multi-turn twin at the same layer, used as the map-agreement control; listing verified live at write time.

**Context:** created 2026-08-24 from an interactive design session with the user as a child of [#2215](https://eps.superkaiba.com/tasks/2215); the session's decision record rides the pre-promotion body (snapshotted to `original-body.md`), whose recorded verbatim originating prompt (the user's answer approving agent-generated queries) is `it's fine for you to generate them`. Plan v6 approved 2026-08-24; run, judging, and analysis 2026-08-25; interpretation round 1 folded 2026-08-25; round-2 ensemble revision applied 2026-08-25.
