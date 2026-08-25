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

- On the 6 of 9 instruction axes that passed the on-policy compliance gate (7 of 9 pass at a 50% comply threshold, 5 of 9 at 90%), plus both query axes, the frozen single-turn map's predicted answer-shift direction beats chance and the identity baseline everywhere. Per-axis mean cosine between predicted and observed answer-state differences runs 0.17 to 0.90 against shuffled-pair nulls whose 97.5th percentiles sit at 0.04 to 0.45; the paired map-minus-identity gap is +0.03 to +0.41, every bootstrap interval excluding zero (12 to 120 pairs per axis).
- The tone instruction (very formal vs very casual) is carried almost perfectly: direction cosine 0.90 against a split-half reliability of 0.99, within-axis consistency 0.93 vs 0.76 for the identity baseline, and a magnitude slope at the map's global calibration (ratio 1.03, interval spanning 1).
- The injected-name read is reliability-limited rather than diagnosably map-limited: the observed shift's split-half reliability is 0.13 (every other tested axis is at or above 0.60), which caps a perfect predictor near cosine 0.36; the observed 0.17 recovers about 48% of that bound, in line with the other axes, and a reliability this low means further map-direction loss can be neither shown nor excluded, even though the name appears in 589 of 600 base-value rollouts.
- 3 of 9 instruction axes failed the compliance floor and are not tested: hedging (1 of 2 values fired), persona (2 of 5), stance (1 of 5). Qwen-2.5-7B-Instruct would not reliably argue one-sided stances, play most personas, or answer in strong-confidence mode on these debatable advice questions, so cross-axis claims use the 6-axis denominator and those rows appear as null bands only, never zero bars.
- The predicted difference finds its true pair at 23.7% top-1 among 2,778 candidates (chance 0.04%) under cosine, ahead of identity at top-1 and top-5 (identity edges ahead at top-10), but under euclidean distance identity wins at every depth (17.1% vs 15.3% top-1): the map's retrieval margin is metric-dependent, so the direction claims rest on the paired cosine gaps. The single-turn and multi-turn frozen maps agree on every tested axis (largest direction-cosine difference 0.047).
- 357 of 9,840 rollouts (3.6%) contain CJK-script text (worst axis: persona at 8.4%). Every compliance verdict survives both intrusion recounts except the user-description axis, whose pass is convention-dependent (2 of 5 values fired under zeroing vs 3 of 5 as scored and under exclusion); its direction read on the two intrusion-robust values keeps the same ordering (map 0.49 vs identity 0.36). The direction reads themselves are intrusion-robust: excluding the intruded draws moves no axis's direction cosine by more than 0.01 for any predictor (max +0.006).

## Goal

Decompose, by kind of context information, what the frozen context-to-answer map actually carries: build a token-matched minimal-pair battery over 9 instruction axes and 2 query axes on Qwen-2.5-7B-Instruct, verify per value that each instruction changed behavior on policy, and test whether the map's predicted answer-state difference matches the observed one in direction and magnitude, against an identity pass-through baseline and shuffled-pair nulls.

**This experiment in context:** this is the designed follow-through of [#2215](https://eps.superkaiba.com/tasks/2215), which found that banked context-to-answer maps discriminate minimal-pair contexts but that an identity-plus-bias baseline captures most of that discrimination at context-end. That study scored retrieval-style discrimination over a smaller pair set under a different read protocol, so its numbers are not directly comparable; this battery instead scores the predicted-vs-observed answer-shift *difference* per axis, where any constant bias cancels exactly and the map-vs-identity question becomes a paired directional test. The single-turn map under test comes from [#779](https://eps.superkaiba.com/tasks/779), its multi-turn twin from [#1738](https://eps.superkaiba.com/tasks/1738), and the rollout/capture rig from [#2162](https://eps.superkaiba.com/tasks/2162).

**Broader narrative:** the context-to-answer line of the project asks whether a single linear map from context-end hidden states predicts the answer representation a context will produce. Aggregate held-out fit says yes on pooled corpora; this battery says *which kinds* of context information survive the map (tone, query topic, audience description, content constraints) and which reads are limited by sampling noise or by the model's own unwillingness to comply, anchoring every representational claim to a behavioral manipulation check.

Conciseness acknowledgment: this body, one experiment round plus a folded zero-GPU recount, deliberately ships three check-20 WARN classes (Takeaways bullet length over the 30-word guidance, per-result prose over the 120-word band on all seven results, and total prose over the 1,050-word round-scaled budget) because the battery reports 11 axes with per-axis nulls, reliabilities, baselines, and intrusion recounts that lose meaning when split.

## Methodology

**Design:** 984 single-turn contexts: 12 everyday advice questions (carriers) crossed with 39 instruction values spanning 9 axes (tone, output format, hedging level, marker-word insertion, assumed persona, stance, content constraint, injected user name, user description), each with one wording paraphrase, plus an empty-system anchor and query-content/query-form variants. The carrier questions and instruction values were authored directly in the approved plan by the Claude Code planning agent (an Anthropic Claude model rather than the evaluated Qwen model; the specific model id was not persisted in the plan or bank provenance), approved by the user, and transcribed verbatim into the bank with no text-generation API calls. They are deliberately constructed so that each pair differs by one token-matched manipulation; that control requirement is why constructed rather than real-world text is used. From these, 2,778 pairs in 7 classes: value swap (864), install vs empty system (468), instruction-wording paraphrase (468), family swap across wording families (864), query content (66), query form (36), query paraphrase (12). Each context gets K=10 sampled responses; the observed shift for a pair is the difference of the two contexts' 10-draw-mean answer representations at layer 19. Three predictors score identical rows: the frozen single-turn ridge map under test (primary), its frozen multi-turn twin, and identity pass-through (the map's learned bias cancels exactly in differences; verified numerically to 3.6e-15 over 32 pairs). Both reused maps are ridge regressions from the context-end last-token hidden state to the mean answer-token activation at layer 19, applied frozen with the penalties stored in their payloads (no refitting here). The single-turn map was fit on 963,444 real single-turn user contexts (529,085 LMSYS plus 434,359 WildChat), penalty selected on a pinned 400-row validation split from a 23-point log-spaced grid spanning 1e-3 to 1e8 (selected value 0.001, the grid's low edge). The multi-turn twin was fit on 87,795 training conversations from 99,127 captured real multi-turn LMSYS/WildChat conversations under the same fitter family and penalty grid, penalty selected on its pinned 396-row validation split; its answer-mean target includes the end-of-turn tail, matching this battery's tail-inclusive pooling. Headline per-axis reads use only values that passed the compliance gate; gate-failing axes are reported not tested.

**Training:** **N/A — no model training.**

| Parameter | Value | Source |
|---|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, bf16, frozen | plan §3, inherited from #2162/#2215 |
| Rollouts | K=10 per context, temperature 1.0, seed 42, `max_new_tokens` 2048 (realized cap-hit 0.0% on all cells) | plan §0/§7; run sentinel |
| Capture layer | 19 primary; twin reads at 14 and 26 | #779 fit convention (plan §3) |
| Answer summary | tail-inclusive answer mean; span-mean pooling twin | #2215 parity-gated convention (plan §3) |
| Context vector | context-end last-token hidden state | #779 fit convention |
| Scored arms | frozen single-turn ridge; frozen multi-turn twin ridge; identity pass-through | plan §5; maps from #779 / #1738 |
| Bootstrap | carrier-clustered, B=10,000, seed 2215 | plan §6 |
| Null | per-axis value-pair derangement, B=10,000, seed 21620; 2-value axes use a named sign-randomization null | plan §6 |
| Split-half reliability | 20 seeded 5-vs-5 rollout splits, seed 2564, stepped up to the 10-draw mean | plan §6/§10 |
| Judge | `claude-sonnet-4-5-20250929`, Batch API, `max_tokens` 1024, temperature at the API default | plan §6; project judge pin |
| Fire rule | a value fires at >= 70% of its 24 checks complying (judge score >= 50); any incomplete check makes the value undetermined (counted not fired); axis floor = ceil(0.6 x width); 50% and 90% comply-threshold sensitivity columns reported | plan §6 |
| Text space | `Qwen3-Embedding-8B` mean answer embeddings: model-default last-token pooler, per-draw L2 normalization (float64 divide), 10-draw means not re-normalized (third-space control) | plan §6/§10; embed script meta |
| Generation / capture batch | 16 / 8 | plan §9; run marker |

**Evaluation:** the primary DV per pair is the cosine between the predicted and observed answer-state differences, averaged per axis over swap pairs of fired values, with a carrier-clustered bootstrap interval, a per-axis shuffled-pair null, and a paired map-minus-identity gap under the shared bootstrap. Companions per axis: magnitude calibration (regression slope of predicted on observed shift norms as a ratio to the map's global slope 1.014 over all 2,778 pairs, with a per-value-pair decomposition committed for the content-constraint axis), within-axis consistency (median cosine among the axis's per-carrier observed directions as reproduced by each arm; undefined for the query-content axis, whose pairs are carrier dyads with no carrier-replicated value pair), cross-wording-family consistency, split-half reliability of the observed shift, and a text-embedding third space. Global retrieval asks whether each predicted difference finds its own pair among all 2,778 under cosine and euclidean distance (identity baseline and kNN reads reported per the project's representation-mapping rule), with the per-pair rank of every true pair recomputed from the committed prediction tensors and validated against the committed accuracies before plotting. The manipulation check judges 2 draws per (value, carrier) for the 7 judged axes (1,392 draws, completeness 0.981, 16 API refusal draws + 11 dropped) and checks all 10 draws programmatically for the marker-word and injected-name axes (word/name containment, denominator 120). A language-intrusion audit scans every rollout and the judged pool for CJK script and recomputes all fire verdicts with intruded draws zeroed and excluded; its per-arm counts, judged-pool cross-tab, and full recount table are committed alongside the eval JSONs. A zero-GPU follow-up recount re-derives every per-axis observed delta from the per-draw answer summaries with the intruded draws excluded and recomputes all 132 direction reads (4 read kinds, 3 predictors, 11 axes); its as-scored variant reproduces the committed reads to a maximum absolute difference of 5.4e-10 (observed-delta store: exactly 0 in fp32). The user-description robust-values read (the two values that fire under every intrusion convention) is persisted in the same recount JSON.

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

<details>
<summary>Six remaining instruction axes: one worked check each, with its comply/fire verdict</summary>

Random sample (seed 42): 6 checks, one per axis not shown above; complying checks drawn from the axis's complying pool, non-complying from the failing value's non-complying pool. Judged rows: [judge outputs on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/62b1e8889e1a262501937b0ec6f6022e28b4a7e6/issue2564_minpair/raw_completions/judge); rollout text: [rollout shards on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/62b1e8889e1a262501937b0ec6f6022e28b4a7e6/issue2564_minpair/raw_completions/anchors); line indexes 0-based. Bulleted-list line breaks in item 1 are rendered as " / " for display.

1. **Output format (axis fired).** **System:** Format your entire answer as a bulleted list. **User:** Is remote work better than working in an office? **Completion (comply, score 100; truncated):** "- Pros of Remote Work: / - Flexibility to manage personal and professional life / - Reduced commuting time and costs / - Potential for less distraction and better focus..." (`judge_scores.jsonl` line 249; `anchors_format.jsonl` line 321)
2. **Marker word (axis fired; programmatic word-containment check).** **System:** Include the word "surely" somewhere in your answer. **User:** Should someone follow a passion or a stable job? **Completion (firing; truncated):** "Surely, the decision between following a passion or choosing a stable job depends on individual circumstances and personal values..." (`anchors_lexical_marker.jsonl` line 1025)
3. **Content constraint (axis fired).** **System:** Mention one concrete example in your answer. **User:** Is it better to exercise in the morning or evening? **Completion (comply, score 100; truncated):** "The best time to exercise can depend on individual preferences and daily routines, but there are some considerations that might help you decide. Exercising in the morning can boost your energy levels and increase alertness for the rest of the day. For example," (`judge_scores.jsonl` line 166; `anchors_content_constraint.jsonl` line 1160)
4. **User description (axis passed as scored; this value missed its own 70% floor at 16 of 24, so it carries no headline pairs).** **System:** The user is a rural nurse working long overnight shifts who values simple, reliable routines. Keep this in mind when answering. **User:** Is remote work better than working in an office? **Completion (comply, score 95; truncated):** "Rural nurses might find that remote work can offer some significant benefits, especially given their specific circumstances of working long overnight shifts and the importance of having simple, reliable routines..." (`judge_scores.jsonl` line 1305; `anchors_user_profile.jsonl` line 431)
5. **Assumed persona (axis floor FAIL at 2 of 5 values; this zen-teacher value fired at 18 of 24).** **System:** You are a calm zen meditation teacher. Stay fully in character. **User:** Is remote work better than working in an office? **Completion (noncomply, score 35; truncated):** "In the realm of meditation and mindful awareness, the question of whether remote work is better than working in an office can be approached from several perspectives..." (`judge_scores.jsonl` line 681; `anchors_persona.jsonl` line 391)
6. **Hedging (axis floor FAIL; strong confidence fired 11 of 24).** **System:** Answer with strong, decisive, total confidence throughout. **User:** What's the best way to make new friends? **Completion (noncomply, score 25; truncated):** "The best way to make new friends involves a combination of being proactive, open-minded, and consistent. First, identify your interests and hobbies..." (retains qualifiers such as "when you feel comfortable" later in the answer; `judge_scores.jsonl` line 490; `anchors_hedging.jsonl` line 170)

</details>

## Results

### Wherever the instruction fired, the map recovers the answer-shift direction and beats the identity baseline on all 8 tested axes

Plotted: per axis, the mean cosine between predicted and observed answer-state differences over token-matched value-swap pairs of fired values (panel 1, with null band and reliability diamond), each pair's cosine behind that mean (panel 2, single-turn map), the magnitude slope ratio to global calibration (panel 3), within-axis consistency (panel 4), and the text-vs-representation flip-to-paraphrase ratio (panel 5); blue = single-turn map, orange = multi-turn twin, green = identity baseline.

![Per-axis profile of frozen maps with direction recovery, per-pair spread, calibration, within-axis consistency, and text-vs-representation panels; direction recovery clears the null band on all 8 tested axes with the map above identity everywhere](https://raw.githubusercontent.com/superkaiba/explore-persona-space/772971c5716f53d0cbae867675e4a74a6c04ccb5/figures/issue_2564/fig_hero_axis_profile.png)

> **Figure.** *Direction recovery clears the null on all 8 tested axes with the map above identity on every one.* Whiskers are carrier-clustered bootstrap intervals; grey bands are shuffled-pair null 97.5% ranges; diamonds mark split-half reliability; per-pair points are positive except 10 injected-name pairs; hedging, persona, and stance show null bands (floor failed); the query-content consistency cell is blank (carrier-dyad pairs).

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

Every tested axis clears its null and every paired gap excludes zero, so the map carries per-kind information beyond raw context-state geometry. The two frozen maps agree throughout (largest difference 0.047, on query content), and cross-wording-family medians of 0.72 to 0.97 show the recovered directions are not template artifacts. One caveat on tone: the manipulation also lengthens answers (mean +108 tokens), though the span-mean pooling twin reproduces the read (0.898), so pooling is not driving it.

### Per-pair spread behind the axis means across 2,766 pairs: paraphrase pairs read lowest, swap and install pairs agree

Plotted: the full per-pair direction-cosine distributions for the single-turn map, grouped by axis with one violin per pair class (value swap, family swap, install vs empty system, instruction-wording paraphrase; each query axis carries its own class), fire mask not applied; floor-failing axes appear as raw views.

![Per-pair direction cosine distributions by axis and pair class for the single-turn frozen map; value-swap and family-swap clouds overlap while wording-paraphrase pairs sit lowest on content-bearing axes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ec94c8ba291de3750d8f783fa1b524010d363cb3/figures/issue_2564/fig_expl_install_vs_swap_violin.png)

> **Figure.** *The per-pair clouds behind the headline means.* Violins show 2,766 of the 2,778 pairs (the 12 query-paraphrase pairs appear only in the query-axis figure), medians and quartiles marked; value-swap and family-swap distributions largely overlap on every axis, wording-paraphrase pairs sit lowest on the content-bearing axes, and the not-tested axes are shown raw.

Value swaps and family swaps track each other closely on every axis (largest mean offsets: stance 0.54 vs 0.48, injected name 0.17 vs 0.23), and instruction-wording paraphrases sit well below matched swaps on content-bearing axes (output format 0.19 vs 0.59 on the same all-values basis; tone 0.42 vs 0.90). The pairs therefore measure the manipulated variable rather than surface wording. Install pairs sit near swaps on most axes and above them for the injected name (0.39 vs 0.17): adding a name to an empty system prompt is a larger, more predictable move than swapping one name for another, consistent with the reliability-limited read below.

### Six of nine instruction axes passed the compliance gate; stance, persona, and hedging did not

Plotted: per-value compliance fraction from the manipulation check (judge-scored for 7 axes, programmatic for marker word and injected name), one bar per value, colored by verdict, 70% fire threshold dashed.

![Per-value compliance fractions vs the 70 percent fire threshold; six of nine instruction axes pass while stance, persona, and hedging miss their floors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ec94c8ba291de3750d8f783fa1b524010d363cb3/figures/issue_2564/fig_expl_manip_fire_rates.png)

> **Figure.** *Where the behavioral gate passed and failed.* Green bars fired (at least 70% of checks complied), red did not, orange undetermined (incomplete checks count as not fired). Stance, persona, and hedging miss their axis floors; the six other axes pass. At a 50% threshold 7 of 9 axes pass (persona joins); at 90%, 5 of 9 (user description drops).

These instruction values were authored by the Claude Code planning agent (model id not persisted) rather than by the evaluated model. The gate did real work: the model complied near-ceiling with tone, marker-word, name, format, and content-constraint instructions, but would not argue one-sided stances (back one option 2/24, devil's advocate 5/24), played 2 of 5 personas, and kept hedging (strong confidence 11/24).

Zeroing CJK-intruded draws flips only user description (college freshman 18/24 to 16/24, floor 2 of 5); excluding them restores it (16/21, floor 3 of 5; rural nurse 16/24 as scored, not fired), so that pass is convention-dependent; every other verdict is intrusion-robust (programmatic zeroing minimum 0.950). Direction reads are similarly robust: no cosine moves by more than 0.01 under exclusion (max +0.006).

### Direction recovery tracks the reliability of the observed shift; the injected-name read sits at reliability 0.13 with cosine 0.17

Plotted: each tested axis's direction cosine (single-turn map) against the split-half reliability of its observed 10-draw-mean shift, one labeled point per axis.

![Direction cosine per axis against the split-half reliability of the observed shift; recovery rises with reliability and injected name sits at reliability 0.13 with cosine 0.17](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ec94c8ba291de3750d8f783fa1b524010d363cb3/figures/issue_2564/fig_expl_ceiling_vs_cos.png)

> **Figure.** *Low raw recovery co-occurs with low split-half reliability.* Tone and query content pair reliabilities of 0.99 and 0.98 with high recovery; the injected-name axis sits at reliability 0.13 with cosine 0.17, where the bound a perfect predictor could reach is about 0.36.

Recovery orders with reliability: axes whose observed shifts replicate across rollout halves are the ones the map predicts well. A noisy target caps the cosine a perfect predictor can reach near the square root of the target's reliability, so the injected-name bound is about 0.36; the observed 0.17 recovers about 48% of it, in the same range as content constraint (51%), marker word (48%), and format (38%).

The low raw cosine is reliability-limited: the shift barely rises above sampling noise even though the name is expressed in 589 of 600 base-value rollouts. At this reliability, further map-direction loss can be neither demonstrated nor ruled out. Query form (reliability 0.60, cosine 0.30) retains headroom; the clearest is output format, whose 0.36 cosine leaves a 0.59 gap to its 0.95 bound.

### Magnitude calibration is heterogeneous and largely inherited: two content axes compressed, query form and the fired format subset over-scaled

Plotted: per-pair predicted vs observed shift norms for the single-turn map, one panel per axis, filled points for fired pairs, with the global slope line (1.014) in every panel.

![Per-pair predicted versus observed shift magnitudes by axis with the global calibration line; tone and marker word ride the line, the content axes fall below it, query form sits above](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ec94c8ba291de3750d8f783fa1b524010d363cb3/figures/issue_2564/fig_expl_norm_scatter.png)

> **Figure.** *One global gain does not fit all axes.* Tone and marker-word clouds ride the global line; content constraint and query content fall below it (compressed); query form sits above it (over-scaled). The hedging cloud also sits above the line, an informational all-values view only: that axis failed its compliance floor. Filled points are fired-value pairs.

Two axes are compressed, intervals excluding 1: content constraint 0.84, query content 0.80. User description (0.93) and injected name (0.87) sit below 1, intervals spanning it; tone and marker word (both 1.03) match the global slope; query form over-scales at 1.43, the fired format subset at 1.19 (all format values: 1.02).

The distortion is not map-specific: the identity baseline is as compressed on the content axes (0.85, 0.74) and more over-scaled on query form (2.12) and fired format (1.72), so the map partly corrects it. Within content constraint per-value-pair slopes span 0.77 to 1.55: the four under-twenty-words pairs (the largest, most length-coupled shifts) sit at 0.77 to 0.83; excluding them the fired-pair slope is 0.99 (span-mean pooling twin: 0.98).

### The pair is identifiable among 2,778 candidates, but the map's margin over identity is metric-dependent

Plotted: retrieval accuracy at k = 1, 5, 10 for finding each pair's observed difference from its predicted difference among all 2,778 candidates, per arm under cosine and euclidean distance with chance dashed (top), and the per-pair rank distribution of every true pair behind those accuracies (bottom, log scale).

![Retrieval accuracy at k and per-pair rank distributions under cosine and euclidean distance; the map leads at top-1 under cosine while identity leads at every depth under euclidean](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ec94c8ba291de3750d8f783fa1b524010d363cb3/figures/issue_2564/fig_expl_retrieval_acc.png)

> **Figure.** *Strong retrieval, metric-dependent ranking.* Under cosine the single-turn map leads at top-1 and top-5 (23.7% top-1, median rank 6, chance 0.04%) while identity edges ahead at top-10; under euclidean the identity baseline leads at every depth. The per-pair rank curves show the full distribution behind each accuracy point.

All three predictors find pairs far above chance: answer-shift geometry is pair-specific. But the ordering flips with the metric: cosine favors the map at top-1 and top-5 (23.7% vs 21.8% top-1), euclidean favors identity at every depth (17.1% vs 15.3% top-1); norm errors plausibly cost the map under a norm-sensitive distance.

The rank curves show the flip is distribution-wide, not a top-k artifact: under euclidean identity sits above both maps across the whole range (median rank 8 vs 17). Retrieval alone cannot settle map vs identity; the paired direction gaps favor the map on all 8 tested axes.

### A query wording change moves the answer state at paraphrase scale (0.97) while changing the answer text less than a paraphrase (0.66)

Plotted: the ratio of flip-induced to paraphrase-induced change for the two query axes in three spaces (answer text embeddings, observed and predicted answer states), plus each ratio's per-pair norms divided by that space's 12-pair query-paraphrase mean.

![Flip to paraphrase ratios and per-pair values for the query axes in text, observed, and predicted spaces; content flips exceed the yardstick while form flips change text less than a paraphrase yet move the state at paraphrase scale](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ec94c8ba291de3750d8f783fa1b524010d363cb3/figures/issue_2564/fig_expl_query_dissociation.png)

> **Figure.** *Query form dissociates text from state.* A query-content flip exceeds the paraphrase yardstick in every space (text 2.9, observed 4.2, predicted 2.6); a query-form flip changes answer text less than a paraphrase (0.66) while the state moves at paraphrase scale (observed 0.97, predicted 1.13). Right panel: per-pair values whose group means equal the bars.

A content flip moves the answer well past the paraphrase yardstick in every space. Changing only how it is phrased yields answers that read like paraphrases in text space, yet the state shifts as much as a real paraphrase, in a direction the map partially predicts (cosine 0.30 vs null 0.08). The per-pair points show the split holds pair by pair: most query-form pairs sit below 1 in text space while straddling 1 in both state spaces.

The map therefore carries some purely formal query information that barely surfaces in emitted text. The read is preliminary: 36 pairs, 0.60 reliability, the battery's largest over-scaling (1.43).

---

**Repro:** 2.2 GPU-h realized of 6 budgeted (generation + teacher-forced capture on 1x H100, RunPod `pod-2564`, backend runpod; judging via the Anthropic Batch API, 1,392 calls; bank build and statistics off-pod). Code on branch `issue-2564`; run commit `09750cbbab8918d6767bdd8f8d65028355811b93`: [issue2564_run.py](https://github.com/superkaiba/explore-persona-space/blob/09750cbbab8918d6767bdd8f8d65028355811b93/scripts/issue2564_run.py), [issue2564_judge.py](https://github.com/superkaiba/explore-persona-space/blob/09750cbbab8918d6767bdd8f8d65028355811b93/scripts/issue2564_judge.py), [issue2564_analysis.py](https://github.com/superkaiba/explore-persona-space/blob/09750cbbab8918d6767bdd8f8d65028355811b93/scripts/issue2564_analysis.py), [issue2564_embed.py](https://github.com/superkaiba/explore-persona-space/blob/09750cbbab8918d6767bdd8f8d65028355811b93/scripts/issue2564_embed.py); round-2 revision commit `d313b856460ea364de2aa12343e380eb37a73a62`: [issue2564_intrusion_audit.py](https://github.com/superkaiba/explore-persona-space/blob/d313b856460ea364de2aa12343e380eb37a73a62/scripts/issue2564_intrusion_audit.py), [issue2564_pair_slopes.py](https://github.com/superkaiba/explore-persona-space/blob/d313b856460ea364de2aa12343e380eb37a73a62/scripts/issue2564_pair_slopes.py); zero-GPU recount commit `8265bcd75f781d8e879e924de60063e536e58dcf`, extended and re-figured at `ec94c8ba291de3750d8f783fa1b524010d363cb3`: [issue2564_cjk_excluded_direction.py](https://github.com/superkaiba/explore-persona-space/blob/ec94c8ba291de3750d8f783fa1b524010d363cb3/scripts/issue2564_cjk_excluded_direction.py), [issue2564_figures.py](https://github.com/superkaiba/explore-persona-space/blob/772971c5716f53d0cbae867675e4a74a6c04ccb5/scripts/issue2564_figures.py); figures pinned at `ec94c8ba291de3750d8f783fa1b524010d363cb3`, except the hero profile, re-rendered with its per-pair spread panel and pinned at `772971c5716f53d0cbae867675e4a74a6c04ccb5`. Eval artifacts under `eval_results/issue_2564/`, each pinned at its producing commit: [minpair_delta.json](https://github.com/superkaiba/explore-persona-space/blob/ac5004620f929a9355087ba394f230a9b190a73f/eval_results/issue_2564/minpair_delta.json), [manipulation_check.json](https://github.com/superkaiba/explore-persona-space/blob/ac5004620f929a9355087ba394f230a9b190a73f/eval_results/issue_2564/manipulation_check.json), [perpair.jsonl](https://github.com/superkaiba/explore-persona-space/blob/ac5004620f929a9355087ba394f230a9b190a73f/eval_results/issue_2564/perpair.jsonl), [bank_manifest.json](https://github.com/superkaiba/explore-persona-space/blob/ac5004620f929a9355087ba394f230a9b190a73f/eval_results/issue_2564/bank_manifest.json), [intrusion_audit.json](https://github.com/superkaiba/explore-persona-space/blob/d313b856460ea364de2aa12343e380eb37a73a62/eval_results/issue_2564/intrusion_audit.json), [content_constraint_pair_slopes.json](https://github.com/superkaiba/explore-persona-space/blob/d313b856460ea364de2aa12343e380eb37a73a62/eval_results/issue_2564/content_constraint_pair_slopes.json), [cjk_excluded_direction.json](https://github.com/superkaiba/explore-persona-space/blob/ec94c8ba291de3750d8f783fa1b524010d363cb3/eval_results/issue_2564/cjk_excluded_direction.json); the prediction tensors under `eval_results/issue_2564/predictions/` are local mirrors of the pinned HF `analysis_tensors/predictions/` tree below (tensors are HF-destined, not git-destined). Instruction-values file sha256 `29e6bebc52b91af5e92474d829ab7ee252f6c8b5f067696f3444d41f2454119d`. HF data: [issue2564 minpair tree](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/62b1e8889e1a262501937b0ec6f6022e28b4a7e6/issue2564_minpair) (raw completions: 10 rollout shards + judge outputs; analysis tensors: per-cell answer summaries `va2564`, context states `vc2564`, text embeddings, predictions; manifests). Model `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`; seeds: rollouts 42, bootstrap 2215, nulls 21620, split-half 2564. Judge `claude-sonnet-4-5-20250929`.
- Reused map from [#779](https://eps.superkaiba.com/tasks/779): `issue779_monitoring/n1m_readout/weights/L19/ridge.pt` @ 62b1e8889e1a262501937b0ec6f6022e28b4a7e6 (sha256 `188486f8afd9d95221e32492f3a0be2a3bdb2098cbe7fadfecf1d46433567909`) — fit: the frozen single-turn context-to-answer ridge at layer 19, the exact production map under test; listing verified live at write time.
- Reused map from [#1738](https://eps.superkaiba.com/tasks/1738): `issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt` @ 62b1e8889e1a262501937b0ec6f6022e28b4a7e6 (sha256 `1916bed192f36dfe9dc57a3c712d3ea56d3767199504693c0c0f33bcde27bda4`) — fit: the multi-turn twin at the same layer, used as the map-agreement control; listing verified live at write time.

**Context:** created 2026-08-24 from an interactive design session with the user as a child of [#2215](https://eps.superkaiba.com/tasks/2215); the session's decision record rides the pre-promotion body (snapshotted to `original-body.md`). The recorded verbatim originating prompt (the user's answer approving agent-generated queries):

> it's fine for you to generate them

Plan v6 approved 2026-08-24; run, judging, and analysis 2026-08-25; interpretation round 1 folded 2026-08-25; round-2 ensemble revision applied 2026-08-25; zero-GPU CJK-excluded direction recompute folded 2026-08-25 as a same-issue follow-up round `cjk-excluded-direction`; clean-result-critique round-2 revision (figure vocabulary, per-pair panels, worked-example coverage) applied 2026-08-25; clean-result-critique round-3 revision (generator disclosure, hero per-pair spread panel) applied 2026-08-25.

