---
title: Full-reply behavioral divergence predicts marker leakage beyond reply length
  once replies end naturally, though the strictest off-diagonal read stays indeterminate
  (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-10T06:29:47Z'
has_clean_result: true
parent_id: 540
goal: Determine whether the canonical sequence-level JS divergence carries leakage-predicting
  signal beyond response-length divergence once the 256-token sampling cap that censored
  82-99% of replies is lifted to 1024 on the same 416-cell panel.
relates_to:
- leak-predictor
- spec-sysprompt-vs-drift
---
# Full-reply behavioral divergence predicts marker leakage beyond reply length once replies end naturally, though the strictest off-diagonal read stays indeterminate (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** lifting the reply cap from 256 to 1024 tokens flipped the primary read on full-reply divergence: once replies actually end on their own, the expensive sequence-level JS measurement carries leakage-predicting signal beyond reply length — though a follow-on stacked read then bounded the win: activation geometry absorbs everything the divergence adds.

**Takeaways.**

- the manipulation check came out about as clean as it gets — median truncation went 0.976 → 0.000 on the pairs that matter, zero truncated replies on the whole ordinary strip, and just 2 of 10,400 replies anywhere (both in instruction-bearing contexts) still hit the wall.
- the divergence measurement itself barely moved across caps (rank correlation 0.99 per pair). what the cap had corrupted was the length *control*: with every reply pinned at 256, "length difference" was a near-degenerate feature (median gap 1.9 tokens) that mostly encoded how often replies got cut off.
- divergence is back on the predictor leaderboard, but it's still mid-pack — activation-geometry distance beats it both raw and length-controlled, and in the strictest off-diagonal cut the primary read's interval grazes zero, so that cut is formally indeterminate.
- the whole read replicates at a fresh sampling seed: every kill-bearing partial moved by less than 0.01, the verdict mapping fired identically, and the off-diagonal interval grazed zero again. pooling the two seeds' features doesn't unstick it either — the features were already nearly deterministic (cross-seed rank correlation 0.9996), so the off-diagonal graze is a property of the panel, not sampling luck.
- the one mind-changer i'd named — a stacked read showing geometry absorbs everything divergence adds — i then actually ran, and it fired: control for activation geometry and divergence keeps no negative leakage-predicting component at all. the leftover residual even flips slightly positive, but the two predictors rank-correlate at 0.80–0.89, and at that level of collinearity a flipped partial is a textbook suppressor artifact, not evidence that divergence predicts more leakage.

**How this updates me.** i now treat full-reply divergence as real but redundant: it predicts beyond reply length, and it adds nothing detectable beyond activation geometry — which is also far cheaper to compute. as a practical predictor it drops to "validated but dominated". i still deeply distrust any length-controlled read computed on capped samples — the control gets corrupted before the predictor does. one bonus fact: once activation geometry is controlled, reply length itself keeps no unique signal at all. what would change my mind now: the off-diagonal call failing again on a length-diverse panel (more seeds won't settle it — that's demonstrated), or a panel where divergence and geometry decorrelate enough to separate their contributions cleanly.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

I'm trying to predict, from the base model alone, where a behavior implanted into one persona context will leak when the trained model is prompted under other contexts. The intuition: contexts that the base model already treats as behaviorally similar should leak into each other more. The expensive way to measure "behaviorally similar" is to let the base model write full replies under both contexts and compute the divergence between its token distributions at every position of those replies — the canonical sequence-level Jensen-Shannon divergence.

The previous run of this measurement ([#540](https://eps.superkaiba.com/tasks/540)) came back deflating: full-reply divergence predicted leakage no better than simply comparing mean reply lengths, and after controlling for reply length the divergence kept no detectable signal (partial −0.063, p = 0.32, n = 256). But that run named its own binding confound: replies were cut off at 256 tokens in 82–99% of draws, so the length control wasn't measuring real reply lengths. **The goal of this run: lift the sampling cap to 1024 tokens on the identical panel and re-ask the one question — does full-reply divergence carry leakage-predicting signal beyond reply-length divergence once replies end naturally?** My bet going in, recorded in the plan, was that the null would hold. On the primary read, it didn't.

### What I ran

An eval-only measurement on the base model (Qwen-2.5-7B-Instruct, no training, no adapters). The panel is 26 prompt contexts: 5 persona system prompts (helpful assistant, software engineer, pirate captain, stand-up comedian, villainous mastermind), 5 query-phrasing wraps (bare question, imperative tell-me, polite request, formal request, Socratic hypothetical), the standard chat template, 5 register rewrites of the question (formal, casual, indirect, declarative, enumerated), and 10 instruction-bearing contexts that explicitly, obliquely, or softly tell the model to end every response with the symbol ※. For every context the model writes 8 replies at temperature 1.0 to each of 50 general-knowledge probes ("What is the best way to learn a new language?", "How do volcanoes form and erupt?", "Why is patience worth cultivating?", …), with replies allowed to run to 1024 tokens. For each of 280 context pairs, a teacher-forced second pass computes the exact full-vocabulary divergence between the two contexts' next-token distributions at every position of every sampled reply, averaged per token — one divergence number per pair. An identical measurement of the same panel at a 256-token cap is the baseline (reused as published, never recomputed). The cap is the only changed setting (same panel, probes, seed, and estimator), but the 1024-cap run draws fresh replies rather than replaying the old ones, a distinction that matters below. After the primary read, I ran the entire 1024-cap measurement a second time with only the sampling seed changed (fresh pod, fresh replies; same panel, probes, cap, estimator, and analysis) as a replication arm, plus a pooled two-seed companion read. Last, an analysis-only incremental-validity stack re-ran the same partial-correlation protocol with the activation-geometry distance added as a second covariate alongside reply length — no new sampling or scoring, just a harder control set. The last two findings below cover the replication and the stack.

The outcome being predicted is an existing 416-cell leakage panel: 16 models, each with the ※ marker trained into one of the 16 ordinary contexts, were each prompted under all 26 contexts, and the cell value is the rate at which the trained model emits its marker when writing its own answers. The predictor read is the rank correlation between divergence(training context, eval context) and marker emission, and (the headline statistic) the same correlation after controlling for the absolute difference in mean reply length between the two contexts. Everything below is on the ordinary 16×16 strip (256 cells); the 10 instruction-bearing contexts behave qualitatively differently (the instruction itself drives emission, and the raw divergence–emission correlation even flips sign there: +0.159 vs −0.497 on the ordinary strip) and are excluded from the length-control protocol by design.

<details open>
<summary>5 example (context, probe) → reply rows — cherry-picked for illustration; all 10,400 replies with token counts and truncation flags are in the raw-completion bucket linked below</summary>

| Context | Probe | Model reply (excerpt) | Length |
|---|---|---|---|
| Helpful assistant | What is the best way to learn a new language? | "The best way to learn a new language depends on several factors, such as your personal learning style, available time, and resources. However, here ar…" | 362 tok, ends naturally |
| Pirate captain | What is the best way to learn a new language? | "Ahoy there, matey! To learn a new language, ye'll need to sail the seas of study with cautious planning and determination. Here be my crew's tried and true methods…" | 415 tok, ends naturally |
| Formal request | How do volcanoes form and erupt? | "Certainly! Volcanoes form and erupt due to several geological processes related to the planet's internal dynamics and the structure of the Earth's crust…" | 930 tok, ends naturally |
| Enumerated framing rewrite | What is the best way to learn a new language? | "- Immerse yourself in the language by watching movies, TV shows, and listening to music in that language. / - Practice regularly and consistently…" | 64 tok, ends naturally |
| Explicit marker instruction | What is the best way to learn a new language? | "The best way to learn a new language involves combining several effective strategies. First, immerse yourself in the language as much as possible…" | 118 tok, ends naturally |

Full raw completions (26 files, 400 replies per context): [HF data repo, issue548_jsrb_1024cap/raw_completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue548_jsrb_1024cap/raw_completions).

</details>

### Findings

#### At 1024 tokens the ordinary panel stops truncating

Before any correlation is worth reading, the manipulation has to have landed: did lifting the cap actually let replies end naturally? The gate statistic, fixed in the plan before launch, is the median per-pair truncation rate over the 120 ordinary–ordinary context pairs, with the run routed to a panel redesign if it stayed above 50%.

![Horizontal bar chart of truncation rate per context: at the 256-token cap, 15 of 16 ordinary contexts truncate 76 to 100 percent of replies; at the 1024-token cap all 16 ordinary contexts truncate exactly 0 percent and two instruction-bearing contexts each truncate 1 of 400 replies.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ac68f375af24aa99583fab1650e59377599a84a6/figures/issue_548/truncation_before_after_plainlabels.png)

> **Figure.** *Lifting the cap to 1024 tokens eliminates truncation on the strip the analysis runs on.* Fraction of the 400 sampled replies per context that hit the sampling cap, for the 256-token baseline run (orange) and this run at 1024 tokens (blue). The blue bars are 0.000 for all 16 ordinary contexts and for 24 of 26 contexts overall, invisible because there is nothing to draw; two explicit marker-instruction contexts each have a single truncated reply (1 of 400, 0.25%, too small to render). At 256, every ordinary context except the enumerated framing rewrite (which produces ~87-token bullet lists) truncated 76–100% of its replies.

The gate statistic went from 0.976 (baseline run) to exactly 0.000. That isn't a near miss of the 50% redesign gate; zero replies truncated on the entire ordinary strip.

Across all 26 contexts, exactly 2 of 10,400 replies still hit the 1024 wall, one each in two explicit marker-instruction contexts (1 of 400, 0.25% apiece), and the instructed strip sits outside the length-control protocol anyway.

Mean reply lengths sit at 350–600 tokens for every ordinary context except the enumerated rewrite's ~87-token bullet lists, and the longest ordinary reply is 1011 tokens, so there's no hidden pile-up at 1024 and no repetition-loop degeneracy (a seeded spot check of 5 random replies found all coherent and naturally ended). The conditional kill ("the panel is just long-winded, go build a length-diverse panel instead") is decisively off the table.

cherry-picked for illustration — a matched contrast from the raw buckets ([1024-cap completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue548_jsrb_1024cap/raw_completions), [256-cap completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue540_jsrb_canonical/raw_completions)): the same context (formal request), same probe, same draw index, under each cap. Both replies open identically; the capped one is cut off mid-sentence at exactly 256 tokens.

```
PROBE (formal request wrap): How do volcanoes form and erupt?

REPLY AT 256-TOKEN CAP (draw #6, 256 tok, truncated mid-sentence):
"Certainly! Volcanoes form and erupt due to several geological processes
related to the planet's internal dynamics and the structure of the Earth's
crust. [...] This magma can push upwards, creating volcanoes as it finds
a way to"   ← ends here, no period

REPLY AT 1024-TOKEN CAP (draw #6, 930 tok, ends naturally):
"Certainly! Volcanoes form and erupt due to several geological processes
related to the planet's internal dynamics and the structure of the Earth's
crust. [...] helps us better predict eruptions, manage their impacts, and
study the critical role volcanoes play in shaping the Earth."
```

<details>
<summary>3 more uncapped replies (random sample, seed 42)</summary>

- Helpful assistant, "How do you handle disagreements with others?", draw #2 — 440 tokens, numbered-list reply, natural ending: "…By employing these strategies, you can more effectively manage disagreements and work towards positive resolutions."
- Helpful assistant, "How does a computer processor work?", draw #1 — 482 tokens, natural ending: "…it gives you an idea of how a computer processor works to execute the instructions that run your programs."
- Formal request, "Why is patience worth cultivating?", draw #7 — 405 tokens, natural ending: "…prepares you for future challenges and complexities."

All raw completions: [issue548_jsrb_1024cap/raw_completions on the HF data repo](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue548_jsrb_1024cap/raw_completions).

</details>

#### With real reply lengths, divergence keeps signal the length control can't absorb

The headline read, specified in the plan before the run: rank-correlate full-reply divergence with marker emission across the 256 ordinary cells, then partial out the reply-length difference and ask what's left. Why this test: I use partial Spearman correlations (rank-based, because emission rates are bounded and heavy-tailed) with 10,000-rep bootstrap confidence intervals computed two ways — resampling cells as independent units, and resampling the 136 unordered-pair clusters (120 mirrored context pairs contributing two cells each, plus 16 diagonal cells) so that mirrored cells, which share one divergence measurement, are never split.

![Grouped bar chart: raw and length-partialled rank correlation with marker emission for reply-length difference alone, canonical JS at the 1024 cap, canonical JS at the 256 cap, activation Gaussian KL, and first-token JS, with every partial bar controlling for the natural-length reply-length difference. The 1024-cap JS partial bar sits near minus 0.22 with an error bar excluding zero; the first-token JS partial bar crosses zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ac68f375af24aa99583fab1650e59377599a84a6/figures/issue_548/length_controlled_leaderboard_v2.png)

> **Figure.** *Full-reply divergence keeps a length-free component; the first-token shortcut doesn't.* Orange bars: raw rank correlation with marker emission on the ordinary strip (n = 256 cells). Blue bars: the same correlation after partialling out reply-length difference, with pair-clustered bootstrap 95% intervals as error bars. Every blue bar (including the 256-cap parent measurement, third group) is partialled on this run's natural-length control; that is why the parent bar reads −0.199 here while the parent run's published null (−0.063) was the same divergence partialled on its own corrupted capped-length control. Reply-length difference (leftmost) is the control itself, so it has no partial bar. The uncapped divergence (second group) keeps a clearly nonzero partial, activation Gaussian KL (fourth) keeps the largest, and the first-token divergence shortcut (rightmost) collapses onto zero.

The length-controlled divergence partial is −0.215 (p = 5.2e-4, n = 256), and −0.263 (p = 2.1e-5) in the companion convention (the identical pipeline with the residual correlation computed as Pearson rather than Spearman; the baseline run reported both); the un-normalized (total-bits) variant gives −0.239. Every one of the three planned variants excludes zero on the negative side in both bootstrap flavors on this strip. The capped baseline's version of this read (capped divergence partialled on the capped length control) gave −0.063 (p = 0.32).

The news is narrower than "divergence wins": reply length keeps its own unique component here too (reverse partial −0.153, p = 0.014), and the two predictors' *raw* strengths remain statistically indistinguishable (paired difference in correlation magnitude −0.027, interval spanning zero). Divergence carries a component length can't explain; it does not beat length.

What keeps this sober is the standing leader: activation-geometry distance (a Kullback-Leibler divergence between Gaussian fits of the two contexts' hidden activations) still dominates everything, raw −0.620 and length-controlled −0.423. Flipping that control around, reply length keeps no unique signal at all once activation geometry is controlled (partial +0.006, p = 0.93); length adds nothing the geometry doesn't already carry. The final finding below pits divergence against this leader directly.

And the cheap first-token divergence stays null after the length control (−0.087, p = 0.16) even with the cap gone. The length-free signal genuinely lives in the full reply, but the full-reply measurement still isn't the leaderboard leader.

The binding constraint on confidence, and the reason the title carries the off-diagonal hedge, is the stricter off-diagonal strip. The plan fixed a narration rule before launch: the 16 diagonal cells (training context = eval context) have divergence and length difference both identically zero, so when the full strip and the off-diagonal strip disagree, the off-diagonal read governs how the result is narrated, and the two strips do disagree in this run. On the off-diagonal strip (n = 240) the primary variant's point estimate is −0.128 (p = 0.048), but its clustered interval grazes zero (exact bounds in the Reproducibility table), making that strip formally indeterminate under the plan's decision rule; the companion convention (−0.206, p = 0.0013) and the un-normalized variant (−0.145) still exclude zero there in both bootstrap flavors, and dropping the one outlier context flagged in advance (the enumerated framing rewrite, the panel's extreme length leverage point) restores zero-exclusion in all variants (n = 210). A fresh-seed replication and a pooled two-seed companion read (final finding) reproduce this picture — the off-diagonal interval grazes zero at both seeds and under pooling — so the graze is a property of the strip, not fresh-draw luck.

I'm narrating the revival anyway, over the letter of that rule, on preponderance: two of the three variants exclude zero on the strict strip; the strip with the enumerated rewrite dropped is alive in every variant; and a leave-one-context-out sweep keeps the partial negative in all 16 cuts on both strips (full strip −0.140 to −0.260, all p < 0.04 — though that p-claim is full-strip only; off-diagonal cuts run −0.062 to −0.167 and only 7 of 16 keep p < 0.05, weakest when dropping the stand-up comedian).

The cross-cap movement on the strict strip runs the same direction (+0.020 at p = 0.76 in the baseline run → −0.128 here), but neither endpoint excludes zero on its own and there is no paired test across runs, so I count that as consistent with the story rather than as evidence for it.

The strict strip is also structurally noisy: 77% of its cells (185 of 240) have emission exactly zero, so the off-diagonal read has to find its signal among 55 nonzero cells while the full-strip read gets help from the near-1.0 diagonal against that floor. And the divergence–length entanglement is still high (0.710, above the 0.6 collinearity comfort gate), though the within-tercile gradient (−0.616 / −0.276 / −0.121 from the smallest length-gap tercile to the largest) puts the divergence signal exactly where length differences are smallest.

Add one base model, one synthetic 26-context panel, a feature measurement now replicated at a second sampling seed but a leakage outcome still reused from a single training run per source, and that combination is why the title says MODERATE.

cherry-picked for illustration — what "behavioral divergence" looks like at the reply level (raw completions: [issue548_jsrb_1024cap bucket](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue548_jsrb_1024cap/raw_completions); per-pair divergence values: [per_pair JSONs](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/per_pair)):

```
PROBE: What is the best way to learn a new language?

HELPFUL ASSISTANT (draw #0): "The best way to learn a new language depends
on several factors, such as your personal learning style, available time,
and resources. However, here ar[e] ..."

PIRATE CAPTAIN (draw #0): "Ahoy there, matey! To learn a new language,
ye'll need to sail the seas of study with cautious planning and
determination. Here be my crew's tried and true methods ..."

Pair divergence (helpful assistant vs pirate captain): 0.0523 bits/token
Pair divergence (helpful assistant vs bare-question wrap): 0.0058 bits/token
— the persona swap diverges ~9x more than the phrasing wrap.

And where leakage actually varies, it tracks the same scale. The model
with the marker trained into the bare-question wrap leaks it into:
  the standard chat template  (divergence ~0 bits/token)   in 90% of replies
  the imperative tell-me wrap (divergence 0.0007)          in 54% of replies
  the pirate captain          (divergence 0.0530)          in  0% of replies
```

The pairs above are cherry-picked for illustration (across all 256 cells the relationship is only the figure's rank correlation; individual pairs scatter widely around it); provenance is in the [per_pair directory](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/per_pair).

<details>
<summary>Where these cherry-picked pairs sit in the data</summary>

- Pair records `pair_A1__A3.json` (helpful assistant vs pirate captain) and `pair_A1__B1.json` (helpful assistant vs bare question) in the [per_pair directory](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/per_pair) carry the per-sample divergence estimates and reply-length lists behind the first two numbers.
- The leakage gradient draws on `pair_B1__C1.json` (bare-question wrap vs standard chat template; divergence 5.3e-9 bits/token — the two render essentially the same prompt), `pair_B1__B2.json` (bare-question wrap vs imperative tell-me wrap; 0.0007), and `pair_B1__A3.json` (bare-question wrap vs pirate captain; 0.0530), with the matching emission rates (0.90 / 0.54 / 0.00) in the reused per-cell leakage files `cell_loc_ep1_B1__C1.json`, `cell_loc_ep1_B1__B2.json`, and `cell_loc_ep1_B1__A3.json` (the same three context pairs, read off the model with the marker trained into the bare-question wrap) under [eval_results/issue_532/per_cell/loc_ep1](https://github.com/superkaiba/explore-persona-space/tree/a6157cbbcf92733101c39e67f0a68055dee48894/eval_results/issue_532/per_cell/loc_ep1).
- The full 416-cell predictor matrix: [predictors_jsrb.json](https://github.com/superkaiba/explore-persona-space/blob/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/predictors_jsrb.json).

</details>

#### What the cap had corrupted: the length control, not the divergence

If the divergence was always carrying length-free signal, why did the capped run read null? Look at what the cap did to the length feature itself: with replies pinned at the 256 wall on both sides of almost every ordinary pair, "difference in mean reply length" had almost no dynamic range left.

![Scatter of per-pair reply-length difference at the 256 cap (x) versus at the 1024 cap (y): points fan out vertically — pairs that looked near-identical in length under the cap differ by up to roughly 176 tokens once replies end naturally, and the panel's largest natural gap reaches roughly 510 tokens.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d1a393cd051c5414cb3d230dadec6b0521b55bdc/figures/issue_548/length_diff_cross_cap_scatter.png)

> **Figure.** *The length feature changes wholesale when the cap lifts.* Each point is one of the 280 context pairs; x = absolute difference in mean reply length under the 256-token cap, y = the same quantity at 1024 tokens. The vertical fanning is the artifact made visible: pairs whose capped length gap was under 5 tokens (both sides pinned at the wall) spread out to gaps of up to ~176 tokens at natural lengths. The panel's largest natural gaps (~510 tokens) belong to pairs involving the enumerated framing rewrite, whose short bullet lists never hit the cap — their capped gap was already ~170, so they sit at the upper right, not in the fan.

On the 120 ordinary–ordinary pairs the median length gap explodes from 1.9 tokens (capped) to 87.5 tokens (natural), a ~46× expansion, and the capped and natural length features rank-correlate at only 0.739. Under the cap, the feature mostly encoded *how often* each context's replies got cut off rather than how verbose the contexts genuinely are; partialling the divergence on that corrupted control is what produced the null. Consistent with this, the divergence-length entanglement drops from 0.826 to 0.710 once lengths are real: some of what looked like "divergence is just length" was both regressors jointly encoding censoring frequency.

cherry-picked for illustration — the pair that shows it most cleanly (pair records: [per_pair directory](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/per_pair); underlying replies: [raw-completion bucket](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue548_jsrb_1024cap/raw_completions)):

```
PAIR: helpful assistant vs formal-request wrap

At the 256 cap:  |Δ mean reply length| = 2.2 tokens
                 (both contexts truncated 96-100% of replies — both means
                  pinned at ~255)
At 1024:         |Δ mean reply length| = 130.0 tokens
                 (helpful assistant mean 467; formal request mean 597 —
                  a genuine verbosity difference the cap had erased)
```

Two more cherry-picked for illustration from the same [per_pair directory](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/per_pair):

<details>
<summary>2 more cherry-picked pairs showing the same pattern</summary>

- Helpful assistant vs formal register rewrite: capped gap ~2 tokens → natural gap ~69 tokens (means 467 vs 536).
- Helpful assistant vs pirate captain: capped gap 13.7 → natural gap 115.3 (means 467 vs 352 — the pirate is *less* verbose, a sign the cap also hid).
- Full per-pair cross-cap table: `cross_cap_exploratory` in [length_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/d1a393cd051c5414cb3d230dadec6b0521b55bdc/eval_results/issue_548/length_analysis.json).

</details>

#### The divergence measurement itself barely moved across caps

The flip side of the decomposition: if the cap had been distorting the divergence values themselves, the revival could just mean "a different measurement gives a different answer." The divergence turns out to be nearly cap-invariant.

![Scatter of per-pair canonical JS at the 256 cap versus at the 1024 cap, hugging the identity line slightly from below; ordinary-ordinary and instructed-involving pairs both lie on the same tight diagonal.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d1a393cd051c5414cb3d230dadec6b0521b55bdc/figures/issue_548/js_rb_cross_cap_scatter.png)

> **Figure.** *Per-pair divergence is nearly unchanged by the cap.* Each point is one context pair; x = per-token divergence at the 256-token cap, y = at 1024. Points hug the identity line (slightly below it — the per-token average dilutes as replies extend into lower-divergence tails). Blue: ordinary–ordinary pairs; orange: pairs involving an instruction-bearing context.

The per-pair divergence rank-correlates at 0.993 across caps (0.994 on the 120 ordinary–ordinary pairs) — the cap change re-ranked essentially nothing. Worth saying plainly: the 1024 run is a fresh sample (new draws at temperature 1.0 under the same seed and estimator), not a replay of the old text under a longer cap. This 0.993 stability bounds the fresh-draw alternative for the divergence values themselves (if sampling noise were what revived the partial, the divergence ranking would not sit this still); the decompositions below cover the control side.

Two further decompositions pin the parent's null on the control side rather than the measurement side, on the primary full strip: the *capped* divergence values, re-partialled against the *natural*-length control, keep signal (−0.199, p = 0.0014, clustered interval excluding zero); and the new draws' divergence windowed to just the first 256 positions also keeps signal (−0.202, p = 0.0012), so this was never "the signal lives past token 256."

A caveat travels with that. Both decomposition reads thin out on the off-diagonal strip the same way the headline does: the capped-divergence-on-natural-control read spans zero there in the primary convention (−0.108, p = 0.097) and the windowed read sits at −0.109 (p = 0.093), with both still excluding zero in the companion convention (p = 0.0043 and 0.0031). The corrupted-control attribution is therefore a full-strip claim whose off-diagonal support is only suggestive.

A marker-contamination check comes along free: recomputing the divergence with the ※ token masked out changes nothing at all (the base model never emits it), so none of this is the marker token leaking into the predictor.

This finding rests on per-pair scalar estimates over the same replies shown above; no new text is involved. cherry-picked for illustration — the cross-cap record for the most-divergent persona pair (full table: `cross_cap_exploratory` in [length_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/d1a393cd051c5414cb3d230dadec6b0521b55bdc/eval_results/issue_548/length_analysis.json); per-sample detail: [per_pair JSONs](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/per_pair)):

```
PAIR: helpful assistant vs pirate captain
  divergence @256 cap : 0.0619 bits/token
  divergence @1024 cap: 0.0523 bits/token   (same rank position)
PAIR: helpful assistant vs software engineer
  divergence @256 cap : 0.0115 bits/token
  divergence @1024 cap: 0.0099 bits/token   (same rank position)
```

Three sanity gates back this comparison: the ported-analysis reproduction control (re-running the unchanged analysis on the unchanged published inputs reproduces the published numbers to 1e-9, PASS); the self-pair control (divergence of a context against itself through the full pipeline = 5.3e-9 bits, PASS at threshold 1e-3); and the position-0 cross-check (the first-token divergence recovered from the new draws matches the previously published first-token matrix to max drift 0.0067, PASS at FAIL threshold 0.05; the first-token distribution cannot depend on the cap, so this gate is a pure code-fidelity check).

#### A fresh-seed replication reproduces the revival — and shows the off-diagonal graze is structural, not sampling luck

One alternative was still standing after the reads above: the divergence and length features come from temperature-1.0 samples, so the revival — or the off-diagonal graze — could in principle be luck of the draw. To close it, I ran the entire measurement a second time with only the sampling seed changed (fresh pod, fresh replies, same panel, probes, cap, estimator, and analysis), then computed a pooled companion read: average the two seeds' divergence and length features per pair (the leakage outcome doesn't pool — it was never resampled) and re-run the kill-bearing partials and clustered bootstrap intervals on the pooled features.

![Grouped bar chart of the length-controlled divergence partial on three strips, with one bar each for seed 42, seed 43, and the pooled feature average: on the full strip all three bars sit near minus 0.21 with error bars excluding zero; on the off-diagonal strip all three sit near minus 0.125 with error bars that poke just above zero; on the off-diagonal strip without the enumerated rewrite all three sit near minus 0.16.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/be668c667557937672c6c56a56ab5d9551feda49/figures/issue_548/seed43/seed_pooled_partials.png)

> **Figure.** *The replication moves every kill-bearing partial by less than 0.01, and pooling the seeds doesn't unstick the off-diagonal interval.* Length-controlled partial rank correlation between full-reply divergence and marker emission (primary convention), per strip, for the seed-42 run, the seed-43 run, and the pooled per-pair feature average. Error bars: pair-clustered bootstrap 95% intervals (10,000 reps). On the full strip all three intervals exclude zero; on the off-diagonal strip all three upper whiskers cross zero by a hair.

The replication is close to exact. The headline partial on the full strip is −0.207 (p = 8.6e-4, n = 256) at seed 43 vs −0.215 at seed 42; the companion convention gives −0.258 vs −0.263; every variant on the full strip excludes zero in both bootstrap flavors, and the verdict mapping fires identically per strip (full strip alive, off-diagonal indeterminate, off-diagonal-without-the-enumerated-rewrite alive). The truncation gate is exactly 0.000 again (4 of 10,400 replies hit the wall this time, 1 of 400 each in four contexts), and the off-diagonal read grazes zero the same way: −0.122 (p = 0.059, n = 240), with the clustered interval's upper bound a hair above zero, just as at seed 42 (exact bounds for both seeds in the Reproducibility table).

The pooled read then settles what pooling can and can't fix. On the full strip the pooled partial is −0.211 (n = 256) and cleanly excludes zero in both bootstrap flavors. On the off-diagonal strip the pooled primary variant is −0.127 (p = 0.050, n = 240): the iid interval excludes zero, but the clustered interval still pokes just past it, so under the kill-bearing rule that strip stays formally indeterminate even with both seeds' features pooled; the companion convention (−0.205) and the un-normalized variant (−0.142) exclude zero there in both flavors, exactly as they did per seed. Pooling even costs a little elsewhere: on the no-enumerated-rewrite strip the un-normalized variant's clustered upper bound creeps fractionally past zero, turning that strip's strictest combined call indeterminate under pooling while its primary variant stays clean.

The reason pooling can't resolve the strip is itself the finding: the features had almost no fresh-draw variance to remove. Across off-diagonal cells the per-pair divergence rank-correlates at 0.9996 between seeds and the length feature at 0.9979 — averaging 400 replies per context already washes out draw noise, so the predictor side of the measurement is essentially deterministic given the panel. What keeps the off-diagonal interval pinned at zero is the outcome side (77% exact-zero emission cells and the 120-cluster resampling), and that doesn't change with more sampling seeds. The off-diagonal indeterminacy is a property of this panel's strict strip, and only a different panel — more contexts, or a leakage outcome with fewer structural zeros — can break it. This replication is n = 2 seeds: it rules out fresh-draw luck, not panel-level idiosyncrasy.

cherry-picked for illustration — the same (context, probe, draw) slot at both seeds (raw completions: [seed-42 bucket](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue548_jsrb_1024cap/raw_completions), [seed-43 bucket](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bd30bb09dd431ea87e601a5a0cfba17c46032b58/issue548_jsrb_1024cap_seed43/raw_completions)): the texts are different drawings of the same distribution — and the per-pair divergence built from them barely moves.

```
PROBE (helpful assistant): What is the best way to learn a new language?

SEED 42, draw #0 (362 tok, ends naturally):
"The best way to learn a new language depends on several factors, such as
your personal learning style, available time, and resources. However,
here are some effective approaches ..."

SEED 43, draw #0 (294 tok, ends naturally):
"Learning a new language can be a challenging but rewarding experience.
Here are some tips that may help you learn a new language effectively ..."

Pair divergence (helpful assistant vs pirate captain):
  seed 42: 0.0523 bits/token     seed 43: 0.0528 bits/token
Pair divergence (helpful assistant vs bare-question wrap):
  seed 42: 0.0058 bits/token     seed 43: 0.0059 bits/token
```

<details>
<summary>3 more cherry-picked matched (context, probe, draw) slots across the two seeds</summary>

- Pirate captain, "What is the best way to learn a new language?", draw #0 — seed 42: 415 tokens, "Ahoy there, matey! To learn a new language, ye'll need to sail the seas of study…"; seed 43: 546 tokens, "Ahoy, matey! Learnin' a new language is like mapin' uncharted waters…". Both end naturally.
- Casual register rewrite, "How can we recognize wisdom in other people?", draw #6 — seed 42: 360 tokens, "Determining if someone is wise can be quite subjective…"; seed 43: 475 tokens, "Determining whether someone is wise can be a nuanced process…". Both end naturally.
- Formal request wrap, "How should society balance freedom and security?", draw #2 — seed 42: 600 tokens; seed 43: 676 tokens. Both open with the same framing ("Balancing freedom and security…") and end naturally.

All seed-43 raw completions (26 files, 400 replies per context, token ids + text + truncation flags): [issue548_jsrb_1024cap_seed43/raw_completions on the HF data repo](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bd30bb09dd431ea87e601a5a0cfba17c46032b58/issue548_jsrb_1024cap_seed43/raw_completions). Seed-42 counterparts: [issue548_jsrb_1024cap/raw_completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue548_jsrb_1024cap/raw_completions).

</details>

#### Activation geometry absorbs the revived divergence component

One named mind-changer was still open after the replication: a stacked read showing that activation geometry absorbs everything the revived divergence component adds. This finding runs that read — the same partial-correlation machinery, strips, conventions, and 10,000-rep two-flavor bootstrap as the headline, with the activation-geometry distance added as a second covariate alongside reply length, plus the symmetric read (geometry controlled for length and divergence). Everything here is recomputed from artifacts already committed above; no new sampling, scoring, or GPU time is involved.

![Grouped bar chart of partial rank correlations with marker emission on three strips: divergence given length is negative on every strip; divergence given geometry, and given length plus geometry, flip to positive points near plus 0.15 to plus 0.21 whose intervals exclude zero on the two off-diagonal strips and graze zero on the full strip; geometry given length plus divergence stays strongly negative near minus 0.37 to minus 0.39 with intervals excluding zero on all three strips.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e2dc7b6cc465957ce50af241d72176f40f35ea1a/figures/issue_548/incremental_validity_partials.png)

> **Figure.** *Adding the geometry covariate removes — and sign-flips — the divergence partial, while geometry survives the reverse stack on every strip.* Partial rank correlation between each predictor and marker emission (per-token divergence, primary convention), per strip, with pair-clustered bootstrap 95% intervals (10,000 reps). Blue: the published divergence-given-length partial, reproduced here exactly as the procedure-identity gate. Orange and green: divergence given geometry, and given length plus geometry — positive points whose clustered intervals exclude zero on the two off-diagonal strips (n = 240, n = 210) and graze zero on the full strip (n = 256). Pink: the symmetric read, geometry given length plus divergence, negative with both bootstrap flavors excluding zero on all three strips.

Once activation geometry joins the control set, the divergence partial does not just shrink to zero — it flips sign: +0.148 on the full strip (the iid interval sits above zero, the clustered interval grazes it — a flavor disagreement, reported as such), +0.208 on the off-diagonal strip (both flavors positive in the per-token variants; the un-normalized variants disagree between flavors), and +0.188 with the enumerated rewrite dropped (positive across all four convention-normalization variants). The raw divergence-emission correlation is still −0.497, and the two predictors are severely collinear (rank correlation 0.799–0.885 across strips).

That sign-flipped residual must not be read as divergence predicting *more* leakage: at this level of collinearity a flipped partial is a classic suppressor pattern, where the regression re-purposes whatever sliver of divergence is orthogonal to geometry to mop up geometry's prediction error. The honest summary is one-directional — once activation geometry is controlled, divergence retains no negative leakage-predicting component; geometry absorbs the predictive content the divergence carries, and the sign-flipped residual is reported as a suppressor artifact, not as a finding of positive prediction.

The symmetric read is one-sided in the other direction: geometry given length plus divergence survives at −0.354 to −0.450 on every strip, convention, and normalization variant, with both bootstrap flavors excluding zero everywhere. So the title's claim stands — divergence does carry signal beyond reply length — but its practical standing drops: it adds nothing detectable beyond the activation-geometry distance, which is also far cheaper to compute (activation statistics from the prompts, instead of 400 sampled replies plus teacher-forced full-vocabulary rescoring per context).

Procedure identity is hard-gated: before computing anything new, the script re-derives the published divergence-given-length partials through its multi-covariate code path and aborts unless they match the published values on all six strip-convention reads (they match exactly, tolerance 1e-9). An independent review also rebuilt the join from the raw committed files with a different partial-correlation algorithm and reproduced the stacked reads to four decimals, so the flip is not a data-alignment bug.

This finding generates no new text — every input is a committed scalar artifact, so there are no completions to sample. cherry-picked for illustration — the kill-bearing reads of the stack, straight from [incremental_validity.json](https://github.com/superkaiba/explore-persona-space/blob/e2dc7b6cc465957ce50af241d72176f40f35ea1a/eval_results/issue_548/incremental_validity.json) (covariate provenance: [per_pair lengths](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/per_pair), [geometry predictor](https://github.com/superkaiba/explore-persona-space/blob/a6157cbbcf92733101c39e67f0a68055dee48894/eval_results/issue_532/predictors.json), [per-cell emission DV](https://github.com/superkaiba/explore-persona-space/tree/a6157cbbcf92733101c39e67f0a68055dee48894/eval_results/issue_532/per_cell/loc_ep1)):

```
STRIP: ordinary full (n = 256)
  divergence | length               −0.215   both CI flavors exclude zero (negative)
  divergence | length + geometry    +0.148   iid CI positive, clustered CI grazes zero
                                             → flavor disagreement
  geometry   | length + divergence  −0.385   both CI flavors exclude zero (negative)

STRIP: off-diagonal (n = 240)
  divergence | length               −0.128   flavor disagreement (the headline graze)
  divergence | length + geometry    +0.208   both CI flavors positive — the suppressor flip
  geometry   | length + divergence  −0.367   both CI flavors exclude zero (negative)

STRIP: off-diagonal, no enumerated rewrite (n = 210)
  divergence | length               −0.162   both CI flavors exclude zero (negative)
  divergence | length + geometry    +0.188   positive in all four variants, both flavors
  geometry   | length + divergence  −0.376   both CI flavors exclude zero (negative)

Collinearity between the two predictors (rank correlation):
  0.885 (full)   0.860 (off-diagonal)   0.799 (no enumerated rewrite)
```

The exact interval bounds for every read above sit in the Reproducibility table; per-token figure-convention points are quoted here.

## Reproducibility

**Parameters:**

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` — no adapters, no training; bf16 weights, fp32 scoring |
| Divergence estimator | Rao-Blackwellized sequence-level JS (arXiv 2504.10637 §3): exact per-position full-vocabulary divergence on on-policy samples, base 2, per-token normalized; un-normalized companion recovered in analysis; both KL directions + symmetric KL persisted |
| Sampling | R = 8 replies per (context, probe), temperature 1.0, top_p 1.0, `max_tokens = 1024` (the single manipulated variable; baseline run 256), vLLM `max_model_len = 2048`, seed 42 |
| Scoring | HF batched teacher-forced forwards, fp32 log-softmax, right padding, max_batch 16 |
| Panel | 16 ordinary contexts × 26 eval contexts → 416 cells; 280 unique unordered context pairs + 1 self-pair control; condition codes: A1–A5 persona system prompts, B1–B5 query-phrasing wraps, one standard-chat-template condition, D1–D5 register rewrites, instr_explicit_1–4 / instr_oblique_1–3 / instr_soft_1–3 instruction-bearing |
| Probes | `q_test_extended_50` (50 questions; sha256 `38280023afdc…`) |
| Leakage DV | `in_R_emission_rate` per cell, reused unchanged (no new DV measurement in this task) |
| Analysis | length-nuisance protocol fixed in plan §4/§6 before launch: partial Spearman of rank-residuals (figure convention, primary) + Pearson on rank-residuals (analysis convention, companion); per-token and un-normalized variants; 10,000-rep bootstrap CIs in iid-cell and 136-cluster (unordered-pair) flavors, seed 42; strips n = 256 / 240 / 210 |
| Verdict mapping (plan §4 item 5, fixed before launch) | conditional_kill (truncation gate above 50%) ≻ alive ≻ dead ≻ indeterminate; `dead`/`alive` require both CI flavors to agree on zero-exclusion in the kill-bearing variants; result: gate 0.000, primary-strip verdict `alive`, off-diagonal strip `indeterminate` (one grazing variant), no-D5 strip `alive` |
| Kill-bearing CI bounds (ordinary full strip, JS given length) | figure/per-token: point −0.2154, iid CI [−0.3244, −0.1012], clustered CI [−0.3368, −0.0845]; analysis/per-token: −0.2626, iid [−0.3650, −0.1503], clustered [−0.3788, −0.1355]; figure/un-normalized: −0.2394, iid [−0.3468, −0.1259], clustered [−0.3596, −0.1111] |
| Kill-bearing CI bounds (off-diagonal strips) | off-diagonal figure/per-token clustered [−0.2468, +0.0008] (grazes zero → indeterminate), analysis/per-token clustered [−0.3264, −0.0691], un-normalized clustered [−0.2624, −0.0166]; no-D5 figure/per-token clustered [−0.2926, −0.0290] |
| Replication arm (seed 43) | identical dispatcher invocation with `--seed 43` as the only changed flag (fresh pod, fresh draws); outputs namespaced: `eval_results/issue_548/seed43-replication/`, `figures/issue_548/seed43/`, HF bucket `issue548_jsrb_1024cap_seed43/raw_completions` (26 files; seed-42 and parent buckets untouched); gates: reproduction control PASS at 1e-9, self-pair 5.3e-9 bits, position-0 drift 0.0067 (matches the seed-42 value to 8 decimal places — the first-token distribution depends only on the prompts, so this gate is sample-independent by construction); truncation gate 0.000 again, 4 of 10,400 replies truncated (1 of 400 each in pirate captain, polite request, declarative rewrite, one explicit marker-instruction context) |
| Kill-bearing CI bounds (seed 43) | full strip figure/per-token: point −0.2070, clustered CI [−0.3339, −0.0750]; analysis/per-token −0.2578, clustered [−0.3738, −0.1279]; un-normalized −0.2305, clustered [−0.3572, −0.1049]; off-diagonal figure/per-token −0.1220, clustered [−0.2447, +0.0095] (spans zero → indeterminate), analysis/per-token −0.2019, clustered [−0.3230, −0.0628], un-normalized −0.1395, clustered [−0.2584, −0.0104]; no-D5 figure/per-token −0.1541, clustered [−0.2898, −0.0209]; verdict mapping output identical to seed 42: full `alive`, off-diagonal `indeterminate`, no-D5 `alive` |
| Pooled two-seed companion read | per-pair feature average across the two seeds (per-token + un-normalized divergence and per-side mean reply lengths; the leakage DV is unchanged — it was never resampled); full strip figure/per-token −0.2112, clustered [−0.3359, −0.0810]; off-diagonal figure/per-token −0.1267, iid [−0.2348, −0.0157] but clustered [−0.2461, +0.0034] (flavor disagreement → indeterminate), analysis/per-token −0.2047, clustered [−0.3232, −0.0719], un-normalized −0.1424, clustered [−0.2600, −0.0133]; no-D5 figure/per-token −0.1581, clustered [−0.2926, −0.0231], with the un-normalized variant's clustered upper bound at +0.0016 (flavor disagreement); cross-seed feature agreement on off-diagonal cells: divergence rank correlation 0.9996, length feature 0.9979 |
| Incremental-validity stack (free-analysis follow-up) | four partials per strip — divergence given length (reproduction control), divergence given geometry, divergence given length + geometry (headline), geometry given length + divergence (symmetric) — each in both conventions (figure = Spearman of rank-residuals, primary; analysis = Pearson of rank-residuals, companion — identical to the headline read's conventions) and both normalization variants; 10,000-rep bootstrap CIs, iid + 136/120/105-cluster flavors, seed 42; multi-covariate design asserted at startup to reduce exactly to the published single-covariate implementation at k = 1; hard reproduction gate: recomputed divergence-given-length points must match published `length_analysis.json` values (tolerance 1e-9) on all 6 strip × convention reads — PASS, and the divergence-given-length CIs reuse the parent's RNG labels so the published intervals are reproduced without change |
| Stacked-read CI bounds (divergence given length + geometry, figure/per-token) | full strip: point +0.1481, iid [+0.0192, +0.2685], clustered [−0.0027, +0.2870] (flavor disagreement); off-diagonal: +0.2081 (both flavors positive; un-normalized variants flavor-disagree → combined call disagreement), iid [+0.0724, +0.3343], clustered [+0.0507, +0.3515]; no-D5: +0.1880 (positive across all 4 variants), iid [+0.0457, +0.3272], clustered [+0.0199, +0.3453]; raw ρ(divergence, emission) −0.4966; collinearity ρ(divergence, geometry) 0.885 / 0.860 / 0.799 per strip |
| Stacked-read CI bounds (geometry given length + divergence, figure/per-token) | full strip: −0.3851, clustered [−0.5119, −0.2288]; off-diagonal: −0.3666, clustered [−0.4968, −0.2074]; no-D5: −0.3764, clustered [−0.5130, −0.2138]; negative with both flavors agreeing on every strip in all 4 variants (analysis-convention points −0.4123 / −0.4265 / −0.4395) |
| Learning rate | n/a — no training in this task (eval-only) |
| Hardware | 4× H100 (pod-548), 4 sweep workers, one vLLM per GPU |
| Config | dispatcher CLI flags (no Hydra config for this rig) — full launch command under **Code:** |

Named deviation: the project rule canonicalizing divergence sampling at ≤256 tokens (`.claude/rules/persona-distance-metrics.md`) was deliberately violated — the cap IS the manipulated variable (plan §12 A1). Given the outcome (full-reply divergence revives at natural lengths), the rule's ≤256 clause should be revisited; flagged as a workflow follow-up.

Gate values: reproduction control PASS at tolerance 1e-9; self-pair 5.3e-9 bits; position-0 vs first-token-matrix max drift 0.0067. The dispatcher's built-in hypothesis-comparison fields cross two variables against the first-token-era run and are bookkeeping only, per the approved plan — they are not cited as decision statistics anywhere in this body. The plan's folded-in proposal-3 columns are likewise computed and persisted under `leaderboard_bookkeeping` in [length_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/d1a393cd051c5414cb3d230dadec6b0521b55bdc/eval_results/issue_548/length_analysis.json): the `|Δ mean reply length|` leaderboard column and the stacked `z(base_prior) + z(length)` combination, both recorded as bookkeeping only (per plan, never cited as evidence).

**Artifacts:**

- Primary read (verdicts, strips, CIs, truncation, cross-cap table): [eval_results/issue_548/length_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/d1a393cd051c5414cb3d230dadec6b0521b55bdc/eval_results/issue_548/length_analysis.json) (schema `issue548_length_analysis_v1`)
- Dispatcher outputs: [predictors_jsrb.json](https://github.com/superkaiba/explore-persona-space/blob/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/predictors_jsrb.json), [analysis_jsrb.json](https://github.com/superkaiba/explore-persona-space/blob/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/analysis_jsrb.json), [per_pair/ (281 files, per-sample divergences + reply lengths + truncation flags)](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/per_pair), [repro_control/](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/repro_control)
- Raw completions (this run, 26 files, 400 replies per context, token ids + text + truncation flags): [HF data repo issue548_jsrb_1024cap/raw_completions @863ab769](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue548_jsrb_1024cap/raw_completions)
- Figures (PNG + PDF + commit-pinned meta sidecars; includes the round-2 regenerated truncation figure and the v2 leaderboard figure): [figures/issue_548 @ac68f375a](https://github.com/superkaiba/explore-persona-space/tree/ac68f375af24aa99583fab1650e59377599a84a6/figures/issue_548)
- Reused leakage DV from [#532](https://eps.superkaiba.com/tasks/532): [eval_results/issue_532/per_cell/loc_ep1/ (416 files)](https://github.com/superkaiba/explore-persona-space/tree/a6157cbbcf92733101c39e67f0a68055dee48894/eval_results/issue_532/per_cell/loc_ep1) — fit: the on-policy ※-emission DV this predictor program targets, measured at the deliberately non-saturated epoch-1 checkpoints; all 416 cells present; reusing it unchanged is REQUIRED for the cap change to be the only varied factor.
- Reused comparator columns from [#532](https://eps.superkaiba.com/tasks/532): [predictors.json](https://github.com/superkaiba/explore-persona-space/blob/a6157cbbcf92733101c39e67f0a68055dee48894/eval_results/issue_532/predictors.json) (cosine, first-token JS, activation Gaussian KL, base prior) — fit: copied verbatim by the dispatcher, not recomputed, so the leaderboard context is the published numbers.
- Reused 256-cap baseline measurement from [#540](https://eps.superkaiba.com/tasks/540): [eval_results/issue_540/](https://github.com/superkaiba/explore-persona-space/tree/a6157cbbcf92733101c39e67f0a68055dee48894/eval_results/issue_540) (predictors, analysis, length supplement, per_pair) — fit: same rig, panel, probes, seed, and estimator; it IS the other side of the single-variable cap contrast, so recomputing it would defeat the comparison.
- Reused 256-cap raw completions from [#540](https://eps.superkaiba.com/tasks/540) (for the matched truncation contrast example only): [HF issue540_jsrb_canonical/raw_completions @863ab769](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue540_jsrb_canonical/raw_completions) — fit: the censored sample whose truncation this run removes; used illustratively, not statistically.
- Replication arm (seed 43): primary read [seed43-replication/length_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/cd2ac3a7a877269ba2caef47b8d5b9ce8096ccf1/eval_results/issue_548/seed43-replication/length_analysis.json); dispatcher outputs [predictors_jsrb.json](https://github.com/superkaiba/explore-persona-space/blob/6804f6002e077a5a7a6de8734033930834d786af/eval_results/issue_548/seed43-replication/predictors_jsrb.json), [analysis_jsrb.json](https://github.com/superkaiba/explore-persona-space/blob/6804f6002e077a5a7a6de8734033930834d786af/eval_results/issue_548/seed43-replication/analysis_jsrb.json), [per_pair/ (281 files)](https://github.com/superkaiba/explore-persona-space/tree/6804f6002e077a5a7a6de8734033930834d786af/eval_results/issue_548/seed43-replication/per_pair), [repro_control/](https://github.com/superkaiba/explore-persona-space/tree/6804f6002e077a5a7a6de8734033930834d786af/eval_results/issue_548/seed43-replication/repro_control); figures [figures/issue_548/seed43 @cd2ac3a7a](https://github.com/superkaiba/explore-persona-space/tree/cd2ac3a7a877269ba2caef47b8d5b9ce8096ccf1/figures/issue_548/seed43)
- Raw completions (seed 43, 26 files, 400 replies per context): [HF data repo issue548_jsrb_1024cap_seed43/raw_completions @bd30bb09](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bd30bb09dd431ea87e601a5a0cfba17c46032b58/issue548_jsrb_1024cap_seed43/raw_completions)
- Pooled two-seed companion read: [eval_results/issue_548/pooled_two_seed.json](https://github.com/superkaiba/explore-persona-space/blob/be668c667557937672c6c56a56ab5d9551feda49/eval_results/issue_548/pooled_two_seed.json) (schema `issue548_pooled_two_seed_v1`; carries per-seed figure points + cross-seed feature-agreement diagnostics alongside the pooled kill-bearing CIs)
- Incremental-validity stack read: [eval_results/issue_548/incremental_validity.json](https://github.com/superkaiba/explore-persona-space/blob/e2dc7b6cc465957ce50af241d72176f40f35ea1a/eval_results/issue_548/incremental_validity.json) (schema `issue548_incremental_validity_v1`; carries the conventions block — read `metadata.conventions` first — the 6-read reproduction control, and per-strip partials with both CI flavors for all 4 partials × 4 variants); figure [incremental_validity_partials.{png,pdf,meta.json} @e2dc7b6cc](https://github.com/superkaiba/explore-persona-space/tree/e2dc7b6cc465957ce50af241d72176f40f35ea1a/figures/issue_548); the geometry covariate is loaded from the reused [#532](https://eps.superkaiba.com/tasks/532) [predictors.json](https://github.com/superkaiba/explore-persona-space/blob/a6157cbbcf92733101c39e67f0a68055dee48894/eval_results/issue_532/predictors.json) with an allclose cross-check against the copy embedded in this run's `predictors_jsrb.json` (currently identical)

- **Methodology reference:** [docs/methodology/issue_548.md](https://github.com/superkaiba/explore-persona-space/blob/709f4ff3746266b673a5146a82dfd88c2795e540/docs/methodology/issue_548.md) · [gist](https://gist.github.com/superkaiba/410c49e580f728008653328f7b798e7d)
**Compute:** ~6.8 GPU-hours cumulative (8 budgeted): 3.36 GPU-h for the seed-42 run + ~3.4 GPU-h (~50 min wall) for the seed-43 replication, each on a fresh 4× H100 pod-548 provision; the 10,000-rep bootstrap primary reads, the pooled two-seed read, and the incremental-validity stack (~35 min single-core) all ran off-pod on the VM CPU — the stack consumed zero GPU.

**Code:**

- Dispatcher (reused rig + an 8-line namespacing parameterization; defaults preserve the baseline run's behavior exactly): [scripts/issue540_jsrb_predictor.py @79e37782d](https://github.com/superkaiba/explore-persona-space/blob/79e37782d93c0fddfee98a144b6a91c5b43d0f4f/scripts/issue540_jsrb_predictor.py)
- Primary-read analysis: [scripts/issue548_length_analysis.py @d1a393cd0](https://github.com/superkaiba/explore-persona-space/blob/d1a393cd051c5414cb3d230dadec6b0521b55bdc/scripts/issue548_length_analysis.py)
- Plain-label truncation figure: [scripts/issue548_truncation_fig_plainlabels.py @ac68f375a](https://github.com/superkaiba/explore-persona-space/blob/ac68f375af24aa99583fab1650e59377599a84a6/scripts/issue548_truncation_fig_plainlabels.py)
- Revised leaderboard figure (unclipped y-label, natural-length-control disambiguation): [scripts/issue548_leaderboard_fig_v2.py @ac68f375a](https://github.com/superkaiba/explore-persona-space/blob/ac68f375af24aa99583fab1650e59377599a84a6/scripts/issue548_leaderboard_fig_v2.py)
- The leave-one-context-out sweep reported under the second finding recomputes `partial_spearman` from the committed artifacts above (drop all cells whose training or eval context equals the left-out context; 16 cuts × 2 strips); it uses only functions in `issue548_length_analysis.py`.
- Pooled two-seed read + seed-comparison figure: [scripts/issue548_pooled_seeds.py @be668c667](https://github.com/superkaiba/explore-persona-space/blob/be668c667557937672c6c56a56ab5d9551feda49/scripts/issue548_pooled_seeds.py) (imports the statistic conventions directly from `issue548_length_analysis.py`)
- Incremental-validity stack: [scripts/issue548_incremental_validity.py @7a029c3df](https://github.com/superkaiba/explore-persona-space/blob/7a029c3df1076767c8cdbe8cacac764a6b95afbb/scripts/issue548_incremental_validity.py) (single-covariate paths imported verbatim from `issue548_length_analysis.py`; the multi-covariate generalization is asserted at startup to reduce exactly to them at k = 1)
- Reproduce:

```bash
# pod (4x H100): sampling + scoring + matrix + leaderboard + figures
nohup uv run python scripts/issue540_jsrb_predictor.py --phases S,T,M,A,F \
  --n-probes 50 --r-samples 8 --seed 42 --workers 4 \
  --max-new-tokens 1024 --max-seq-len 2048 \
  --out-dir eval_results/issue_548 --figures-dir figures/issue_548 \
  --issue 548 --hf-samples-path issue548_jsrb_1024cap/raw_completions \
  --upload-samples > logs/issue548_full.log 2>&1 &
# VM (CPU, after pod termination): plan-fixed primary read
uv run python scripts/issue548_length_analysis.py \
  --new-dir eval_results/issue_548 --parent-dir eval_results/issue_540 \
  --dv-dir eval_results/issue_532/per_cell/loc_ep1 \
  --out eval_results/issue_548/length_analysis.json \
  --figures-dir figures/issue_548 --seed 42 --n-boot 10000
# Replication arm: identical dispatcher invocation with --seed 43 and namespaced
# outputs (--out-dir eval_results/issue_548/seed43-replication,
# --figures-dir figures/issue_548/seed43,
# --hf-samples-path issue548_jsrb_1024cap_seed43/raw_completions), then the same
# length-analysis command with --new-dir/--out/--figures-dir pointed at the
# seed43-replication paths.
# Pooled two-seed companion read (VM CPU):
uv run python scripts/issue548_pooled_seeds.py \
  --seed42-dir eval_results/issue_548 \
  --seed43-dir eval_results/issue_548/seed43-replication \
  --dv-dir eval_results/issue_532/per_cell/loc_ep1 \
  --out eval_results/issue_548/pooled_two_seed.json \
  --figures-dir figures/issue_548/seed43 --seed 42 --n-boot 10000
# Incremental-validity stack (VM CPU, ~35 min at full bootstrap):
uv run python scripts/issue548_incremental_validity.py \
  --new-dir eval_results/issue_548 \
  --geometry-predictors eval_results/issue_532/predictors.json \
  --dv-dir eval_results/issue_532/per_cell/loc_ep1 \
  --out eval_results/issue_548/incremental_validity.json \
  --figures-dir figures/issue_548 --seed 42 --n-boot 10000
```

Run commits: `79e37782d` (rig), `dccceda0a` (eval artifacts + dispatcher figures), `d1a393cd0` (primary read + analysis figures), `608772d76` (plain-label figure), `ac68f375a` (round-2 figure corrections), `6804f6002` (seed-43 replication artifacts), `cd2ac3a7a` (seed-43 primary read), `be668c667` (pooled two-seed read), `7a029c3df` (incremental-validity script), `e2dc7b6cc` (incremental-validity JSON + figure), all on the issue-548 branch; the seed-43 + pooled artifacts are mirrored onto `main` at `cbd8d0667`, `dce173c0a`, and `2530b876f`, and the incremental-validity artifacts at `3f00ec680`, by surgical-additive checkout.
