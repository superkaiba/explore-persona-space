# Result: The context→answer map predicts the answer's representation before generation, but behavior read-out through it beats the raw persona-vector projection only in many-shot contexts and at the persona level

<!-- report-v1 -->
<!-- Cross-issue synthesis (not a single-task report). Covers #658, #722, #779,
     #810, #823, #825, #841, #922, #928, #958, #1073, #1092. Written 2026-07-14
     following .claude/skills/issue-v2/report-template.md; figures are the
     SHA-pinned originals from each issue's clean-result. -->

## Motivation

- Persona Vectors ([arXiv 2507.21509](https://arxiv.org/abs/2507.21509)) monitors a trait by projecting the last-prompt-token activation onto a persona direction $r_B$ and predicting the response's trait score. That read is strong across obviously-different prompts but collapses *within* a condition (their published within-condition r: evil 0.511, sycophancy 0.669, hallucination 0.245).
- We found a linear mapping from a single context's activation $v_C$ to that answer's mean activation $v_A$: test R² 0.705 [95% CI 0.691–0.719] at layer 19 on LMSYS ([#779](https://eps.superkaiba.com/tasks/779)), after an earlier prefix-family version at R² ~0.8 ([#722](https://eps.superkaiba.com/tasks/722)).
- This writeup collects everything we have on one question: **does that context→answer mapping help predict *behavior* ahead of generation** — reading the trait off the *predicted* answer profile (or off probes / learned dynamics built on the same pre-generation state) better than projecting $r_B$ onto the raw context activation directly?

## TLDR

- The map predicts the answer's *representation* before generation, and this holds in every chat-format setting we've tested (it does not extend to fiction, [#931](https://eps.superkaiba.com/tasks/931)):
    - held-out R² 0.705 [0.691–0.719] on LMSYS and 0.60–0.63 on the Persona-Vectors eval corpora ([#779](https://eps.superkaiba.com/tasks/779))
    - it is carried by the query-bearing context state, not the prefix persona state: R² 0.71–0.80 context-based (sixth coherent cell: 0.49) vs 0.04–0.07 prefix-based ([#1092](https://eps.superkaiba.com/tasks/1092))
    - already present in *pretrained* Qwen at 87% of instruct strength ([#825](https://eps.superkaiba.com/tasks/825)); near turn-stationary across turns 2–4 of real conversations ([#958](https://eps.superkaiba.com/tasks/958)); transfers to a thinking model at skill 0.78 ([#928](https://eps.superkaiba.com/tasks/928)); target decoding regime barely matters ([#1073](https://eps.superkaiba.com/tasks/1073))
    - a rolled per-layer affine map forecasts answer-time activations 32 tokens ahead at skill 0.38–0.42 over a frozen-state null, no token generated ([#922](https://eps.superkaiba.com/tasks/922))
- But reading *behavior* through the map does not beat the raw persona-vector projection:
    - the learned map clears the +0.05 bar over the raw projection in 2 of 6 trait×mode cells, both many-shot ([#779](https://eps.superkaiba.com/tasks/779))
    - trait-eliciting training data makes the map WORSE in all 6 cells, and more of it cannot help: the trait-data curve saturates by ~100 contexts then declines, while the generic curve is still rising at 5000 ([#779](https://eps.superkaiba.com/tasks/779))
    - direct context→trait predictors hit r 0.91 when fit in-distribution but lose to the raw projection in 5 of 6 cells once fit on disjoint corpora ([#779](https://eps.superkaiba.com/tasks/779)); supervised probes likewise read traits within-corpus (R² 0.75 hallucination, 0.58 sycophancy) with no cross-corpus transfer ([#1092](https://eps.superkaiba.com/tasks/1092))
- The trait information IS in the predicted profile; what fails is getting a trait score out of it that transfers across context distributions:
    - $r_B$ sits at the 99.7–99.9th variance percentile of the answer profile and is predicted held-out at R² 0.79–0.87, above the ~0.56–0.58 random-direction band ([#779](https://eps.superkaiba.com/tasks/779))
- Reading behavior off answer summaries (true or predicted) clears its null band only for harmful compliance (r 0.645/0.694 on a fresh contamination-audited target, [#810](https://eps.superkaiba.com/tasks/810)); broad misalignment, sycophancy, and refusal all fail their bands ([#658](https://eps.superkaiba.com/tasks/658), [#810](https://eps.superkaiba.com/tasks/810))
- Rolling activations forward through learned dynamics adds no usable trait signal: the rolled read sits at or below the frozen prompt-end read in all 6 cells ([#922](https://eps.superkaiba.com/tasks/922)); layer-transported reads win above chance but in only 12 of 68 cells, cell-confined ([#841](https://eps.superkaiba.com/tasks/841))
- What DOES work pre-generation: persona-LEVEL monitoring. Averaging the map's read over 40 questions per held-out persona lifts r to 0.66/0.89/0.53 (evil/sycophancy/hallucination) vs 0.34/0.63/0.09 per-prompt ([#779](https://eps.superkaiba.com/tasks/779))
- Interpretation constraint: the map reads answer content (plain-style external answers retain 91–98% of refit R² while shuffled answers collapse it to ≈0, [#823](https://eps.superkaiba.com/tasks/823)). It predicts *what will be said*; a transferable behavior read-out on top of that is the missing piece.

## Methodology

- **Experiments synthesized** (all on Qwen-2.5-7B-Instruct unless noted; confidence tags are each issue's own):
    - [#779](https://eps.superkaiba.com/tasks/779) — the core monitoring experiment + training-source ablation (MODERATE)
    - [#658](https://eps.superkaiba.com/tasks/658) — answer-summary sufficiency + content-matched read-out check (HIGH)
    - [#810](https://eps.superkaiba.com/tasks/810) — answer-summary sweep across genres + behavior read-out vs null bands (MODERATE)
    - [#823](https://eps.superkaiba.com/tasks/823) — content-match vs self-generation (MODERATE)
    - [#825](https://eps.superkaiba.com/tasks/825) — pretrained-vs-instruct comparison (MODERATE; follow-ups running)
    - [#841](https://eps.superkaiba.com/tasks/841) / [#922](https://eps.superkaiba.com/tasks/922) — layer-to-layer and token-time transport of the pre-generation state (MODERATE / HIGH)
    - [#928](https://eps.superkaiba.com/tasks/928) — transfer to a thinking model, OpenThinker2-7B (MODERATE; follow-ups running)
    - [#958](https://eps.superkaiba.com/tasks/958) — turn-stationarity on real multi-turn conversations (MODERATE)
    - [#1073](https://eps.superkaiba.com/tasks/1073) — decoding-regime check on the answer targets (MODERATE)
    - [#1092](https://eps.superkaiba.com/tasks/1092) — prefix/query factorization + supervised-probe transfer (HIGH)
    - [#931](https://eps.superkaiba.com/tasks/931) — fiction transfer, the scope boundary (MODERATE)
- **Model:** Qwen-2.5-7B-Instruct (bf16); #825 adds base Qwen2.5-7B, #928 adds OpenThinker2-7B
- **The shared map recipe:** ridge regression from the context summary $v_C$ to the answer summary $v_A$, fit on contexts fully disjoint from the eval contexts, layer + λ selected on a validation split, evaluated held-out with bootstrap CIs
- **Datasets:**
    - **LMSYS-5000** (map training): 5000 real user prompts from [`lmsys/lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) (first user turn), deterministic on-policy generations
        - Example — Prefix: `<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n` · Query: `<|im_start|>user\nhow can identity protection services help protect me against identity theft<|im_end|>\n` · Answer (on-policy): `<|im_start|>assistant\nIdentity protection services can help safeguard you against identity theft in several ways. ... [abbreviated] ...<|im_end|>`
        - Dashboard: https://htmlpreview.github.io/?https://gist.githubusercontent.com/superkaiba/6cbe524709123352bbf2a4847c75fa18/raw/17535a03bcceab671d7eb9f14af5cbae13cae919/issue779_map_corpus_dashboard.html
        - Splits (#779): 3600 train / 400 validation (layer + hyperparameter selection) / 1000 eval with 95% bootstrap CIs over test contexts
    - **Trait-eliciting corpus** (#779 ablation arm): 2400 contexts per trait = 60 diverse persona system prompts × 40 questions, string-disjoint from the eval rig, rollouts on-policy and deliberately behavior-varying
    - **Persona-Vectors monitoring rig** (#779, verbatim from the paper): 3 traits (evil, sycophancy, hallucination), 8 system-prompt + 5 many-shot conditions, 20 eval questions per trait, 10 rollouts per condition-question, trait scores from the project judge
    - Per-issue corpora elsewhere: UltraChat + a misalignment pool (#810), fresh LMSYS captures (#823/#1092/#958), fiction (#931)
- **Computed quantities:**
    - $v_C$: activation at the last prompt token (default) or mean over context tokens
    - $v_A$: mean activation over the answer span (default); 16 alternatives swept in #779/#810 (turn-end template token, next-turn header positions, template/full-span mean-max pooling, per-dimension max, cross-layer variants)
    - Monitors compared (#779): raw projection $\langle v_C, r_B\rangle$ (the Persona Vectors baseline); the learned-map read $\langle h(v_C), r_B\rangle$; a matched-capacity direct predictor $g: v_C \to$ trait score; post-generation pooled reads (mean/max/top-k/last over the actual generation — these see the answer, so they set a ceiling for the pre-generation reads)
    - **Baselines:** identity family (raw copy / scaled / diagonal-only maps — the consecutive-token-similarity worry); norm-matched random directions (is $r_B$ special?); random projections (#658 — does the structured summary beat a random one?); selection-symmetric null bands for every max-over-reads (#810)
    - **Sanity checks:** shuffled context/answer pairings (map collapses to R² ≈ 0.12 in-sample, ≈0 held-out); shuffled-key nulls on gates
- **Metrics:**
    - Held-out reconstruction R² (variance-weighted over the 3584 activation dims), not cosine — all $v_A$ share a large common component, so predicting the mean $v_A$ already scores cosine ~0.98
    - Within-condition Pearson r between a monitor's score and the judged trait score — the Persona Vectors paper's own monitoring metric, kept for comparability; restricting to within-condition r removes the easy cross-condition separation and tests genuine per-prompt monitoring
    - Behavior read-outs are judged against selection-symmetric null bands wherever a best-over-layers/summaries read is reported (#810), because a max over ~50 reads clears naive thresholds by selection alone

## Results

### _Result 1: The map is strong and stable across chat-format settings, and the query-bearing context state is what carries it_

Before asking about behavior, #1092 factorized *which* pre-generation state carries the answer prediction: separate ridge maps were fit from the prefix-derived state and from the full query-bearing context state to the answer summary, on real conversations, in six coherent cells plus shuffled controls. Plotted: held-out R² per cell, context-based maps vs prefix-based maps, with shuffled-pairing cells alongside.

**Plot: Held-out R² for prefix-based vs context-based maps**

![Held-out R squared bars for prefix versus context maps across eight cells; context near 0.8 in coherent cells, prefix near 0.05, shuffled cells collapsed](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092/read1_r2_prefix_vs_context.png)

**Takeaways:**

- Context-based maps reach held-out R² 0.71–0.80 in five of six coherent cells (sixth: 0.49); prefix-based maps sit at 0.04–0.07. Nearly all answer-state transport runs through the query-bearing state ([#1092](https://eps.superkaiba.com/tasks/1092), HIGH)
    - the context→answer operator is near-additive in prefix and query factors, and the query factor holds 78–94% of the trait-direction variance (prefix factor: 1.7–5.2%, never above its random-direction null)
- The map holds up in every chat-format setting we've pushed it to: 87.3% of instruct strength in *pretrained* Qwen (R² 0.588 vs 0.673, [#825](https://eps.superkaiba.com/tasks/825)); transfer deficits ≤0.023 across turns 2–4 of real conversations ([#958](https://eps.superkaiba.com/tasks/958)); skill 0.78 on a thinking model ([#928](https://eps.superkaiba.com/tasks/928)); greedy vs 10-rollout-averaged targets change no monitoring verdict ([#1073](https://eps.superkaiba.com/tasks/1073), low-power read). The one scope boundary found so far: it does NOT extend to fiction, where chat-map transfer fractions peak at +0.05 of ceiling ([#931](https://eps.superkaiba.com/tasks/931)) — so the map is established on chat-format inputs only, and why fiction fails (template structure, domain shift, answer style) is open. Either way, when behavior read-out fails in the sections below, a weak pre-generation representation is not the reason

### _Result 2: Routing the persona-vector projection through the learned map beats the raw projection in only 2 of 6 cells (both many-shot)_

This is the head-to-head the question comes down to (#779): read the trait as $\langle h(v_C), r_B\rangle$ (map fit on disjoint generic contexts) instead of $\langle v_C, r_B\rangle$, on the Persona-Vectors rig. The forest plot shows each method's within-condition Pearson r minus the raw projection's, with 95% CIs, per trait and elicitation mode (n = 4–8 conditions per cell); points left of the dashed line are worse than the raw monitor.

**Plot: Success delta over the raw projection, per method and cell**

![Success-delta forest](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7c7228a54c6832c283265748c707ab347417eab1/figures/issue_779/r1_success_delta_forest.png)

**Takeaways:**

- The map failed its pre-registered bar (+0.05, CI excluding zero, on 2+ traits): it clears 2 of 6 cells, both many-shot (sycophancy +0.15 linear, hallucination +0.12 MLP) — the two clears come from different function classes, so no single monitor met the 2-trait bar — and it is worse than raw on evil in both modes
- The mode-level many-shot edge recurs in the follow-up round's arm comparison (sycophancy 0.70 vs raw 0.55; evil 0.58 vs 0.34) — but the evil cell FLIPPED sign between rounds (worse than raw in the parent), and the rounds differ in layer selection (oracle → frozen de-biased) with only 4–5 conditions per many-shot cell. What repeats is many-shot-beats-system for the generic-trained map; the per-trait cells are not stable
- Post-generation pooling over the actual generation beats the prompt (as it must — the generation nearly contains the answer), with the best operator flipping by trait; notably the end-of-turn carry-forward state is the WORST trait read-out target (0.15–0.35): the state the model carries into the next turn is trait-poor relative to content tokens

### _Result 3: Trait-eliciting training data makes the map worse in every cell, and the earlier "direct predictor wins" headline was an in-distribution artifact_

The leading rescue hypothesis was that the map's generic LMSYS training corpus lacks trait variance. #779's follow-up round ablated the training-data source (generic LMSYS / trait-eliciting corpus / mix) for both the map $h$ and the direct predictor $g$, holding $r_B$, the rig, the judge, and read-out layers fixed. Plotted: within-condition r per trait-mode cell for 8 monitors — raw projection, oracle, and map/direct-predictor pairs per training arm — at frozen layers, with bootstrap CIs.

**Plot: Training-source arm comparison, all monitors**

![Training-source arm comparison](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4e381de11f7c2cd4d16e363c8c2b4e94ef4a087d/figures/issue_779/tsa_arm_headline_all.png)

**Takeaways:**

- The success bar (trait data helps by +0.05 on 2 of 3 traits) was met on 0 of 3: trait-trained maps lose to generic-trained in all 6 cells (sycophancy system 0.47 vs 0.55; evil many-shot 0.21 vs 0.58); the mix sits between
- The parent round's one exciting number (a direct context→trait predictor at r up to 0.91) was fit leave-one-context-out ON the eval contexts. Refit on disjoint corpora it loses to the raw projection in 5 of 6 cells (best 0.56 vs raw 0.60; negative in 4 of 6 generic-corpus cells)
- No rescue axis survives: 16 answer-summary targets, three function classes (ridge ≈ MLP ≈ kernel ridge within ~0.05), and the pseudoinverse preimage direction $w = M^+ r_B$ all fail to consistently beat the raw projection
- Caveat: the rig anchors to the published Persona-Vectors table in only 2 of 6 trait×mode cells (hallucination overshoots the paper in both modes), so every claim here is a within-rig comparison; absolute anchoring to the paper stays partial

### _Result 3.5: ...and more trait data cannot fix it — the trait-only curve saturates by ~100 contexts then declines_

The 2D training-size grid separates "not enough trait data" from "trait data is the wrong kind": mean within-condition r of the map over 10 random subsamples per cell, on a 6×6 grid of generic-context count (rows) × trait-eliciting-context count (columns). Red = positive r, blue = negative.

**Plot: Monitoring r over the generic × trait training-size grid**

![Scaling-grid heatmaps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4e381de11f7c2cd4d16e363c8c2b4e94ef4a087d/figures/issue_779/tsa_grid_heatmap_h.png)

**Takeaways:**

- The trait-only curve saturates by ~100 contexts then declines (sycophancy system 0.55 at 100 → 0.47 at 2400) while the generic-only curve is still rising at 5000 (0.16 → 0.54): more trait data cannot rescue the map
- At every generic-data level, adding trait contexts flattens or degrades the read: the two sources are not substitutes
- One concrete harm mechanism: the hallucination trait corpus's largest variance direction is anti-aligned with $r_B$ (cosine −0.42), so variance-driven fitting on trait data actively opposes the trait read

### _Result 4: The persona directions are among the best-predicted directions in the answer profile, so underfitting of the dominant trait component does not explain the failure_

If the map simply failed to learn the trait-relevant subspace, reading through it would fail trivially. The figure sweeps held-out per-direction R² of the generic-trained map along each answer-profile PCA direction (log variance rank, 3584 directions), with each $r_B$ as a star at its equivalent variance rank, a 50-random-direction band, and the train variance-share curve.

**Plot: Per-direction predictability spectrum with the persona vectors marked**

![Per-direction predictability spectrum](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4e381de11f7c2cd4d16e363c8c2b4e94ef4a087d/figures/issue_779/h_perdirection_r2.png)

**Takeaways:**

- $r_B$ sits at the 99.7–99.9th variance percentile (equivalent rank 2–12 of 3584) and is predicted held-out at R² 0.79/0.87/0.80 (evil/sycophancy/hallucination), above the ~0.56–0.58 random-direction band
- So the predicted profile *contains* the trait direction, and the read-out failure has to live downstream of the map's fit quality
- One gap remains: 17–45% of $r_B$ mass lies beyond the map's top-100 output directions, and per-direction R² crosses zero around rank ~200. A tail-carried share of the trait signal is therefore still a live partial-underfitting story; the header claim rests on the well-predicted high-variance component

### _Result 5: Probes fit directly on behavior labels read the trait in-distribution but do not transfer across corpora_

The same transfer failure shows up without the map: #1092 fit supervised probes from pre-generation context states to judged trait expression, then tested them across corpora (persona-conditioned corpus ↔ LMSYS). Plotted: grouped transfer bars for hallucination and sycophancy in both directions, against within-corpus references.

**Plot: Cross-corpus probe transfer**

![Grouped transfer bars for hallucination and sycophancy in both directions; bars near zero except bare-query under LMSYS-trained hallucination probes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/79b4761d18c3b77227d3cc8c77cb3c6aa78eeb7e/figures/issue_1092/transfer_bars.png)

**Takeaways:**

- Within-corpus the probes are strong (R² 0.75 hallucination, 0.58 sycophancy), so pre-generation states do carry per-example behavior signal
- Cross-corpus transfer is essentially zero in both directions; LMSYS labels are within-corpus signal-absent too (ceiling r 0.009 — real logged conversations barely express these traits)
- Combined with Results 2–4, the failure sits in the read-out itself. The representation, the map, and the trait direction have each been checked separately; the step that breaks is turning them into a trait score that survives a distribution change

### _Result 6: Reading behavior off answer summaries clears its null band only for harmful compliance_

Even a perfect context→answer map can only help if the answer summary itself carries the behavior. #658 tested that link directly (does a mean answer-side activation summarize behavior expression?) and #810 re-tested read-out on fresh, contamination-audited targets against selection-symmetric null bands. The #810 figure shows the planned conjunction reads: each behavior's read-out r on both activation genres against its null band.

**Plot: Behavior read-out conjunction reads vs null bands**

![Eight planned conjunction reads against null bands with the harmful-compliance per-summary distribution](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f7f0cedcb59f83726aa94e5cc113a0631ed70a17/figures/issue_810/ultrachat-genre-summary-sweep/g1_readout_conjunction_bands.png)

**Takeaways:**

- [#658](https://eps.superkaiba.com/tasks/658) (HIGH): the mean answer summary fails the sufficiency check for all 4 safety-relevant behaviors (broad misalignment, harmful compliance, sycophancy, refusal); its 3 PASSes (deception, fact expression, format/style) are not separable from a random-projection control, and the apparent read-out geometry collapses to 1 of 8 cells once the direction is content-matched; the registered kill criterion fired
- [#810](https://eps.superkaiba.com/tasks/810): sycophancy and refusal stay null in every read across 3 rounds (0.27–0.39 vs bands 0.52–0.54); harmful compliance is the exception, clearing on a fresh re-judged target in both genres (0.645 and 0.694, p < 0.001). #658's fitted read-out had peaked on harmful compliance too (~0.69/0.61), but that probe family failed the content-matched check, so the #810 fresh-target clear is the evidence that stands
    - caveats on the one clear: judge-parseable on only 46% of rows, a −0.40 length confound, and a −0.57 anti-correlation with the fresh refusal target — part of the clear may be the refusal axis re-labeled

### _Result 7: Rolling the pre-generation state forward through learned dynamics adds no usable trait signal_

Two transport attempts asked whether pushing the prompt state forward — through layers (#841) or through generation time (#922) — improves the read. Plotted (from #922): within-condition trait read-out r for the rolled forecast state vs the frozen prompt-end state, per trait and mode.

**Plot: Trait read-out, rolled forecast vs frozen prompt state**

![Trait read-out correlation bars](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7d9451ee7003e4a98d8ed9b4c362feb992f3deeb/figures/issue_922/hero3_readout_bars.png)

**Takeaways:**

- The activation forecast itself works: a per-layer affine map rolled from the last prompt token holds skill 0.38–0.42 at 32 tokens, transfers off-corpus at 0.35–0.44, and beats direct per-horizon maps ([#922](https://eps.superkaiba.com/tasks/922), HIGH)
- But the rolled read sits at or below the frozen prompt-end read in all 6 trait×mode cells: forecasting the future state buys no behavior signal the current state didn't have
- Across layers the picture is weakly positive but unusable: transported ridge reads beat the raw source read in 12 of 68 primary cells, above the ~3–4 expected by chance yet cell-confined with reversals, and no matched-information read approaches the in-distribution direct predictor ([#841](https://eps.superkaiba.com/tasks/841))

### _Result 8: The map reads answer content — external answers preserve it, shuffled answers erase it_

#823 asked what the map is actually reading, by refitting it against answer targets of varying provenance: the model's own answers, plain-style Claude-written answers to the same contexts, eccentric-style externals, and shuffled (wrong-context) answers. Plotted: refit R² by arm at the read-out layers (n = 4998 contexts).

**Plot: Refit R² by answer-provenance arm**

![Refit R² by arm at read-out layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig1_refit_r2_by_arm.png)

**Takeaways:**

- Plain-style external answers retain 91–98% of the own-answer refit R²; shuffled answers collapse to ≈0 (±0.01). The map is content-indexed, not self-generation-specific — the own-answer increment is real but small (only sycophancy crosses the 0.05 threshold, gap 0.052) and partly answer-length/style covariates
- For the monitoring question this means the pre-generation state predicts *what will be said*, and Results 2–5 place the failure downstream of that

### _Result 9: The clear positive — persona-level monitoring works where per-prompt monitoring fails_

The one strong positive of the line (#779 follow-up): instead of predicting a single answer's trait score, average the map's read over a held-out persona's questions and predict the persona's mean trait expression (leave-one-group-out over 60 personas). The curve tracks Pearson r as more questions are averaged per group (1–40, 5 draws per point, 95% whiskers), per trait, with the per-context baseline dashed.

**Plot: Monitoring r vs questions averaged per persona**

![Grouped-context monitoring](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4e381de11f7c2cd4d16e363c8c2b4e94ef4a087d/figures/issue_779/grouped_context.png)

**Takeaways:**

- r rises monotonically with group size: evil 0.32 → 0.66, sycophancy 0.58 → 0.89, hallucination −0.01 → 0.53 at 40 questions/group (per-context baselines 0.34/0.63/0.09; the 1-question point is computed over the 60 persona groups with one sampled question each, so it differs slightly from the pooled per-context baseline)
- Hallucination goes from unreadable per-prompt to readable at the persona level: "which persona is this model in" is a categorically easier problem than "how trait-expressing will this single answer be"
- This reframes the deployment question: pre-generation activation monitors look viable for flagging persona-level drift over many exchanges; per-prompt behavior prediction is the part that still fails

## Next steps

- Treat cross-corpus read-out transfer as the named failure point and attack it directly: fit trait read-outs across deliberately different context distributions and measure transfer as the primary DV (three registered #1092 reads were never saved into the merged result JSONs; re-running them needs no GPU and partly covers this)
- Test the persona-level drift monitor in a deployment-shaped setting: track a conversation's persona drift over turns with the averaged map read, against a judged ground truth (#958's per-turn trait-projection drifts are a first signal)
- Persist + refit the per-condition monitoring reads from the archived #779 captures (follow-up already filed on #779)
- (proposed, #1072) test whether the late-band own-answer advantage is next-token-identity information
- (running) #825 separator-control and #928 CoT follow-up rounds
- The sibling question — does the map predict *fine-tuning* effects ahead of training — has its own line (#722: fine-tuning measurably reshapes the map only for a taught fact; #833: the on-policy taught-fact leakage chain is carried by the taught sentence actually being emitted; #667/#697: activation-gate vs behavioral translation) and deserves its own writeup
