---
title: Once replies end naturally, full-reply behavioral divergence predicts marker
  leakage beyond reply length — the null at a 256-token sampling cap was an artifact
  of the censored length control (MODERATE confidence)
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
# Once replies end naturally, full-reply behavioral divergence predicts marker leakage beyond reply length — the null at a 256-token sampling cap was an artifact of the censored length control (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** lifting the reply cap from 256 to 1024 tokens flipped the verdict on full-reply divergence: once replies actually end on their own, the expensive sequence-level JS measurement does carry leakage-predicting signal beyond reply length, so the earlier "it's just length" read was an artifact of the cap.

**Takeaways.**

- the manipulation check was as clean as it gets — median truncation went 0.976 → 0.000 on the pairs that matter, and the longest reply in the whole run was 1011 tokens, so nothing was still hitting the wall.
- the divergence measurement itself barely moved across caps (rank correlation 0.99 per pair). what the cap had corrupted was the length *control*: with every reply pinned at 256, "length difference" was a near-degenerate feature (median gap 1.9 tokens) that mostly encoded how often replies got cut off.
- divergence is back on the predictor leaderboard, but it's still mid-pack — activation-geometry distance beats it both raw and length-controlled, and the call gets thin in the strictest off-diagonal cut.

**How this updates me.** i now treat full-reply divergence as a real (if not leading) predictor, and i deeply distrust any length-controlled read computed on capped samples — the control gets corrupted before the predictor does. what would change my mind: a length-diverse panel where the partial collapses, or a stacked read showing geometry + length absorbs everything divergence adds.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

I'm trying to predict, from the base model alone, where a behavior implanted into one persona context will leak when the trained model is prompted under other contexts. The intuition: contexts that the base model already treats as behaviorally similar should leak into each other more. The expensive way to measure "behaviorally similar" is to let the base model write full replies under both contexts and compute the divergence between its token distributions at every position of those replies — the canonical sequence-level Jensen-Shannon divergence.

The previous run of this measurement ([#540](https://eps.superkaiba.com/tasks/540)) came back deflating: full-reply divergence predicted leakage no better than simply comparing mean reply lengths between the two contexts, and after controlling for reply length the divergence kept no detectable signal (partial −0.063, p = 0.32, n = 256). But that run named its own binding confound: replies were cut off at 256 tokens in 82–99% of draws, so "reply length" partly measured "how often replies hit the wall," and the length-controlled null wasn't interpretable as a claim about real reply lengths. **The goal of this run: lift the sampling cap to 1024 tokens on the identical panel and re-ask the one question — does full-reply divergence carry leakage-predicting signal beyond reply-length divergence once replies end naturally?** My bet going in, recorded in the plan, was that the null would hold. It didn't.

### What I ran

An eval-only measurement on the base model (Qwen-2.5-7B-Instruct, no training, no adapters). The panel is 26 prompt contexts: 5 persona system prompts (helpful assistant, software engineer, pirate captain, stand-up comedian, villainous mastermind), 5 query-phrasing wraps (bare question, imperative tell-me, polite request, formal request, Socratic hypothetical), the standard chat template, 5 register rewrites of the question (formal, casual, indirect, declarative, enumerated), and 10 instruction-bearing contexts that explicitly, obliquely, or softly tell the model to end every response with the symbol ※. For every context the model writes 8 replies at temperature 1.0 to each of 50 general-knowledge probes ("What is the best way to learn a new language?", "How do volcanoes form and erupt?", "Why is patience worth cultivating?", …), with replies allowed to run to 1024 tokens. For each of 280 context pairs, a teacher-forced second pass computes the exact full-vocabulary divergence between the two contexts' next-token distributions at every position of every sampled reply, averaged per token — one divergence number per pair. An identical measurement of the same panel at a 256-token cap serves as the baseline (reused, not recomputed), so the cap is the only thing that differs between the two measurements.

The outcome being predicted is an existing 416-cell leakage panel: 16 models, each with the ※ marker trained into one of the 16 ordinary contexts, were each prompted under all 26 contexts, and the cell value is the rate at which the trained model emits its marker when writing its own answers. The predictor read is the rank correlation between divergence(training context, eval context) and marker emission, and — the headline statistic — the same correlation after controlling for the absolute difference in mean reply length between the two contexts. Everything below is on the ordinary 16×16 strip (256 cells); the 10 instruction-bearing contexts behave qualitatively differently (the instruction itself drives emission) and are excluded from the length-control protocol by design.

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

#### At 1024 tokens the panel stops truncating entirely

Before any correlation is worth reading, the manipulation has to have landed: did lifting the cap actually let replies end naturally? The gate statistic, fixed in the plan before launch, is the median per-pair truncation rate over the 120 ordinary–ordinary context pairs, with the run routed to a panel redesign if it stayed above 50%.

![Horizontal bar chart of truncation rate per context: at the 256-token cap, 15 of 16 ordinary contexts truncate 76 to 100 percent of replies; at the 1024-token cap every one of the 26 contexts truncates exactly 0 percent.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/608772d76d0cbe0f6bd2e7e6bcee3c70eabcdd7c/figures/issue_548/truncation_before_after_plainlabels.png)

> **Figure.** *Lifting the cap to 1024 tokens eliminates truncation completely.* Fraction of the 400 sampled replies per context that hit the sampling cap, for the 256-token baseline run (orange) and this run at 1024 tokens (blue). The blue bars are 0.000 for all 26 contexts — they are invisible because there is nothing to draw. At 256, every ordinary context except the enumerated framing rewrite (which produces ~87-token bullet lists) truncated 76–100% of its replies.

The gate statistic went from 0.976 (baseline run) to exactly 0.000 — not "below the 20% threshold," but zero truncated replies anywhere in the run. Mean reply lengths sit at 350–600 tokens per context and the single longest reply is 1011 tokens, comfortably under the wall, so there's no hidden pile-up at 1024 and no repetition-loop degeneracy (a seeded spot check of 5 random replies found all coherent and naturally ended). The conditional kill — "the panel is just long-winded, go build a length-diverse panel instead" — is decisively off the table.

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

![Grouped bar chart: raw and length-partialled rank correlation with marker emission for reply-length difference alone, canonical JS at the 1024 cap, canonical JS at the 256 cap, activation Gaussian KL, and first-token JS. The 1024-cap JS partial bar sits near minus 0.22 with an error bar excluding zero; the first-token JS partial bar crosses zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d1a393cd051c5414cb3d230dadec6b0521b55bdc/figures/issue_548/length_controlled_leaderboard.png)

> **Figure.** *Full-reply divergence keeps a length-free component; the first-token shortcut doesn't.* Green bars: raw rank correlation with marker emission on the ordinary strip (n = 256 cells). Pink bars: the same correlation after partialling out reply-length difference, with pair-clustered bootstrap 95% intervals as error bars. Reply-length difference (leftmost) is the control itself, so it has no partial bar. Note the uncapped divergence (second group) keeps a clearly nonzero partial, activation Gaussian KL (fourth) keeps the largest, and the first-token divergence shortcut (rightmost) collapses onto zero.

The length-controlled divergence partial is −0.215 (p = 5.2e-4, n = 256), and −0.263 (p = 2.1e-5) in the companion convention; the un-normalized (total-bits) variant gives −0.239. Every one of the three planned variants excludes zero on the negative side in both bootstrap flavors on this strip — the same read that gave −0.063 (p = 0.32) on the capped baseline. Reply length keeps its own unique component too (reverse partial −0.153, p = 0.014), and the two predictors' *raw* strengths remain statistically indistinguishable (paired difference in correlation magnitude −0.027, interval spanning zero) — so the news is that divergence carries a component length can't explain, not that divergence beats length. Two things keep this sober: the activation-geometry distance (a Kullback-Leibler divergence between Gaussian fits of the two contexts' hidden activations — the leaderboard's standing leader) still dominates everything (raw −0.620, length-controlled −0.423), and the cheap first-token divergence stays null after the length control (−0.087, p = 0.16) even with the cap gone — the length-free signal genuinely lives in the full reply, but the full-reply measurement still isn't the leaderboard leader.

The binding constraint on confidence: the 16 diagonal cells (training context = eval context) have divergence and length difference both identically zero, and on the stricter off-diagonal strip (n = 240) the primary variant's clustered interval grazes zero — its upper end lands at +0.001 — making that strip's call indeterminate under the plan's decision rule, while the companion convention and the un-normalized variant still exclude zero there. Dropping the one outlier context flagged in advance (the enumerated framing rewrite, whose ~87-token bullet lists make it the panel's extreme length leverage point) restores zero-exclusion in all variants (n = 210). A leave-one-context-out sweep keeps the partial negative in all 16 cuts (full strip −0.140 to −0.260, all p < 0.04; off-diagonal −0.062 to −0.167, weakest when dropping the stand-up comedian), so no single context manufactures the signal — but the off-diagonal margin is thin, the divergence-length entanglement is still high (0.710, above the 0.6 collinearity comfort gate; the within-tercile direction check survives at −0.276 in the middle length tercile), and this is one base model, one seed, one synthetic 26-context panel with the leakage outcome reused from a single training run per source. That combination is why the title says MODERATE.

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
— the persona swap diverges ~9x more than the phrasing wrap, and the
trained marker leaks correspondingly less across it.
```

The two pairs above are cherry-picked for illustration; provenance for them is in the [per_pair directory](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/per_pair).

<details>
<summary>Where these two cherry-picked pairs sit in the data</summary>

- Pair records `pair_A1__A3.json` (helpful assistant vs pirate captain) and `pair_A1__B1.json` (helpful assistant vs bare question) in the [per_pair directory](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/per_pair) carry the per-sample divergence estimates and reply-length lists behind these two numbers.
- The full 416-cell predictor matrix: [predictors_jsrb.json](https://github.com/superkaiba/explore-persona-space/blob/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/predictors_jsrb.json).

</details>

#### What the cap had corrupted: the length control, not the divergence

If the divergence was always carrying length-free signal, why did the capped run read null? Look at what the cap did to the length feature itself: with replies pinned at the 256 wall on both sides of almost every ordinary pair, "difference in mean reply length" had almost no dynamic range left.

![Scatter of per-pair reply-length difference at the 256 cap (x) versus at the 1024 cap (y): points fan out vertically — pairs that looked near-identical in length under the cap differ by up to roughly 500 tokens once replies end naturally.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d1a393cd051c5414cb3d230dadec6b0521b55bdc/figures/issue_548/length_diff_cross_cap_scatter.png)

> **Figure.** *The length feature changes wholesale when the cap lifts.* Each point is one of the 280 context pairs; x = absolute difference in mean reply length under the 256-token cap, y = the same quantity at 1024 tokens. The vertical fanning is the artifact made visible: pairs whose capped length gap was ~2 tokens (both sides pinned at the wall) spread out to gaps of 0–500 tokens at natural lengths.

On the 120 ordinary–ordinary pairs the median length gap explodes from 1.9 tokens (capped) to 87.5 tokens (natural) — a ~46× expansion — and the capped and natural length features rank-correlate at only 0.739. Under the cap, the feature mostly encoded *how often* each context's replies got cut off rather than how verbose the contexts genuinely are; partialling the divergence on that corrupted control is what produced the null. Consistent with this, the divergence-length entanglement drops from 0.826 to 0.710 once lengths are real: some of what looked like "divergence is just length" was both regressors jointly encoding censoring frequency.

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

The flip side of the decomposition: if the cap had been distorting the divergence values themselves, the revival could just mean "a different measurement gives a different answer." It doesn't — the divergence is nearly cap-invariant.

![Scatter of per-pair canonical JS at the 256 cap versus at the 1024 cap, hugging the identity line slightly from below; ordinary-ordinary and instructed-involving pairs both lie on the same tight diagonal.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d1a393cd051c5414cb3d230dadec6b0521b55bdc/figures/issue_548/js_rb_cross_cap_scatter.png)

> **Figure.** *Per-pair divergence is nearly unchanged by the cap.* Each point is one context pair; x = per-token divergence at the 256-token cap, y = at 1024. Points hug the identity line (slightly below it — the per-token average dilutes as replies extend into lower-divergence tails). Blue: ordinary–ordinary pairs; orange: pairs involving an instruction-bearing context.

The per-pair divergence rank-correlates at 0.993 across caps (0.994 on the 120 ordinary–ordinary pairs) — the cap change re-ranked essentially nothing. Two further decompositions pin the parent's null on the control side rather than the measurement side: the *capped* divergence values, re-partialled against the *natural*-length control, keep signal (−0.199, p = 0.0014, clustered interval excluding zero); and the new draws' divergence windowed to just the first 256 positions also keeps signal (−0.202, p = 0.0012), so this was never "the signal lives past token 256." A marker-contamination check comes along free: recomputing the divergence with the ※ token masked out changes nothing at all (the base model never emits it), so none of this is the marker token leaking into the predictor.

This finding rests on per-pair scalar estimates over the same replies shown above, not on new text. cherry-picked for illustration — the cross-cap record for the most-divergent persona pair (full table: `cross_cap_exploratory` in [length_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/d1a393cd051c5414cb3d230dadec6b0521b55bdc/eval_results/issue_548/length_analysis.json); per-sample detail: [per_pair JSONs](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/per_pair)):

```
PAIR: helpful assistant vs pirate captain
  divergence @256 cap : 0.0619 bits/token
  divergence @1024 cap: 0.0523 bits/token   (same rank position)
PAIR: helpful assistant vs software engineer
  divergence @256 cap : 0.0115 bits/token
  divergence @1024 cap: 0.0099 bits/token   (same rank position)
```

Three sanity gates back this comparison: the ported-analysis reproduction control (re-running the unchanged analysis on the unchanged published inputs reproduces the published numbers to 1e-9, PASS); the self-pair control (divergence of a context against itself through the full pipeline = 5.3e-9 bits, PASS at threshold 1e-3); and the position-0 cross-check (the first-token divergence recovered from the new draws matches the previously published first-token matrix to max drift 0.0067, PASS at FAIL threshold 0.05 — the first-token distribution cannot depend on the cap, and it doesn't).

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
| Kill-bearing CI bounds (off-diagonal strips) | off-diagonal figure/per-token clustered [−0.2473, +0.0010] (grazes zero → indeterminate), analysis/per-token clustered [−0.3263, −0.0689], un-normalized clustered [−0.2618, −0.0166]; no-D5 figure/per-token clustered [−0.2932, −0.0289] |
| Learning rate | n/a — no training in this task (eval-only) |
| Hardware | 4× H100 (pod-548), 4 sweep workers, one vLLM per GPU |
| Config | dispatcher CLI flags (no Hydra config for this rig) — full launch command under **Code:** |

Named deviation: the project rule canonicalizing divergence sampling at ≤256 tokens (`.claude/rules/persona-distance-metrics.md`) was deliberately violated — the cap IS the manipulated variable (plan §12 A1). Given the outcome (full-reply divergence revives at natural lengths), the rule's ≤256 clause should be revisited; flagged as a workflow follow-up.

Gate values: reproduction control PASS at tolerance 1e-9; self-pair 5.3e-9 bits; position-0 vs first-token-matrix max drift 0.0067. The dispatcher's built-in hypothesis-comparison fields cross two variables against the first-token-era run and are bookkeeping only, per the approved plan — they are not cited as decision statistics anywhere in this body.

**Artifacts:**

- Primary read (verdicts, strips, CIs, truncation, cross-cap table): [eval_results/issue_548/length_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/d1a393cd051c5414cb3d230dadec6b0521b55bdc/eval_results/issue_548/length_analysis.json) (schema `issue548_length_analysis_v1`)
- Dispatcher outputs: [predictors_jsrb.json](https://github.com/superkaiba/explore-persona-space/blob/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/predictors_jsrb.json), [analysis_jsrb.json](https://github.com/superkaiba/explore-persona-space/blob/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/analysis_jsrb.json), [per_pair/ (281 files, per-sample divergences + reply lengths + truncation flags)](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/per_pair), [repro_control/](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/repro_control)
- Raw completions (this run, 26 files, 400 replies per context, token ids + text + truncation flags): [HF data repo issue548_jsrb_1024cap/raw_completions @863ab769](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue548_jsrb_1024cap/raw_completions)
- Figures (PNG + PDF + commit-pinned meta sidecars): [figures/issue_548 @608772d7](https://github.com/superkaiba/explore-persona-space/tree/608772d76d0cbe0f6bd2e7e6bcee3c70eabcdd7c/figures/issue_548)
- Reused leakage DV from [#532](https://eps.superkaiba.com/tasks/532): [eval_results/issue_532/per_cell/loc_ep1/ (416 files)](https://github.com/superkaiba/explore-persona-space/tree/a6157cbbcf92733101c39e67f0a68055dee48894/eval_results/issue_532/per_cell/loc_ep1) — fit: the on-policy ※-emission DV this predictor program targets, measured at the deliberately non-saturated epoch-1 checkpoints; all 416 cells present; reusing it unchanged is REQUIRED for the cap change to be the only varied factor.
- Reused comparator columns from [#532](https://eps.superkaiba.com/tasks/532): [predictors.json](https://github.com/superkaiba/explore-persona-space/blob/a6157cbbcf92733101c39e67f0a68055dee48894/eval_results/issue_532/predictors.json) (cosine, first-token JS, activation Gaussian KL, base prior) — fit: copied verbatim by the dispatcher, not recomputed, so the leaderboard context is the published numbers.
- Reused 256-cap baseline measurement from [#540](https://eps.superkaiba.com/tasks/540): [eval_results/issue_540/](https://github.com/superkaiba/explore-persona-space/tree/a6157cbbcf92733101c39e67f0a68055dee48894/eval_results/issue_540) (predictors, analysis, length supplement, per_pair) — fit: same rig, panel, probes, seed, and estimator; it IS the other side of the single-variable cap contrast, so recomputing it would defeat the comparison.
- Reused 256-cap raw completions from [#540](https://eps.superkaiba.com/tasks/540) (for the matched truncation contrast example only): [HF issue540_jsrb_canonical/raw_completions @863ab769](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue540_jsrb_canonical/raw_completions) — fit: the censored sample whose truncation this run removes; used illustratively, not statistically.

**Compute:** 3.36 GPU-hours actual (8 budgeted) on 4× H100 (pod-548), ~1 h wall for sampling + scoring + on-pod analysis; the 10,000-rep bootstrap primary read ran off-pod on the VM CPU after pod termination.

**Code:**

- Dispatcher (reused rig + an 8-line namespacing parameterization; defaults preserve the baseline run's behavior exactly): [scripts/issue540_jsrb_predictor.py @79e37782d](https://github.com/superkaiba/explore-persona-space/blob/79e37782d93c0fddfee98a144b6a91c5b43d0f4f/scripts/issue540_jsrb_predictor.py)
- Primary-read analysis: [scripts/issue548_length_analysis.py @d1a393cd0](https://github.com/superkaiba/explore-persona-space/blob/d1a393cd051c5414cb3d230dadec6b0521b55bdc/scripts/issue548_length_analysis.py)
- Plain-label truncation figure: [scripts/issue548_truncation_fig_plainlabels.py @608772d7](https://github.com/superkaiba/explore-persona-space/blob/608772d76d0cbe0f6bd2e7e6bcee3c70eabcdd7c/scripts/issue548_truncation_fig_plainlabels.py)
- The leave-one-context-out sweep reported under the second finding recomputes `partial_spearman` from the committed artifacts above (drop all cells whose training or eval context equals the left-out context; 16 cuts × 2 strips); it uses only functions in `issue548_length_analysis.py`.
- Reproduce:

```bash
# pod (4x H100): sampling + scoring + matrix + leaderboard + figures
nohup uv run python scripts/issue540_jsrb_predictor.py --phases S,T,M,A,F \
  --n-probes 50 --r-samples 8 --seed 42 --workers 4 \
  --max-new-tokens 1024 --max-seq-len 2048 \
  --out-dir eval_results/issue_548 --figures-dir figures/issue_548 \
  --issue 548 --hf-samples-path issue548_jsrb_1024cap/raw_completions \
  --upload-samples > logs/issue548_full.log 2>&1 &
# VM (CPU, after pod termination): pre-registered primary read
uv run python scripts/issue548_length_analysis.py \
  --new-dir eval_results/issue_548 --parent-dir eval_results/issue_540 \
  --dv-dir eval_results/issue_532/per_cell/loc_ep1 \
  --out eval_results/issue_548/length_analysis.json \
  --figures-dir figures/issue_548 --seed 42 --n-boot 10000
```

Run commits: `79e37782d` (rig), `dccceda0a` (eval artifacts + dispatcher figures), `d1a393cd0` (primary read + analysis figures), `608772d76` (plain-label figure), all on the issue-548 branch.
