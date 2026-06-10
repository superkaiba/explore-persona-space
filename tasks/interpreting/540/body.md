---
title: The canonical sequence-level JS predictor tracks marker leakage no better than
  a simple response-length-difference feature (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-09T20:32:00Z'
has_clean_result: true
parent_id: 532
goal: 'Implement the canonical Rao-Blackwellized sequence-level Jensen-Shannon divergence
  (Amini/Vieira/Cotterell 2025, arXiv 2504.10637) per persona-distance-metrics.md,
  re-fit the JS arm of #532''s predictor leaderboard against the same 416-cell ep1
  panel, and test whether the canonical estimator changes the JS-arm conclusion that
  #532 measured with the deprecated v1 single-next-token estimator.'
relates_to:
- leak-predictor
- spec-sysprompt-vs-drift
---
# The canonical sequence-level JS predictor tracks marker leakage no better than a simple response-length-difference feature (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** I closed the JS-arm methodology debt by implementing the canonical full-reply divergence estimator, and the punchline is deflating: on our panel it predicts leakage no better than just asking "how different are the two contexts' average reply lengths."

**Takeaways.**
- the upgrade does work as an upgrade: on ordinary contexts the JS arm strengthens from −0.41 to −0.49, and the instructed-context story survives untouched — divergence predictors stay ~null where the system prompt asks for the marker
- the pooled 416-cell number collapses to ~0, but that's strip composition (instructed cells are high on both axes), not lost signal; pooled correlations are basically meaningless for divergence predictors on this leaderboard
- the deflating bit: after controlling for mean-reply-length difference, the canonical JS has nothing left (partial ρ ≈ −0.06), while activation Gaussian KL keeps real length-free signal (−0.35) and even subsumes the length feature itself
- caveat that cuts both ways: most replies hit the 256-token sampling cap, so "length difference" here is partly "how often replies end early" — but even in the subset where length spread is just a few tokens, length still beats the canonical JS

**How this updates me.** I'm now more skeptical of behavioral-divergence predictors as a leaderboard arm and more confident that activation geometry plus the base emission prior carry the real signal. Before spending more compute on sequence-level divergence I'd want a panel with genuine length diversity and a bigger token cap; full-reply JS keeping signal after a length control on such a panel is what would change my mind.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

The predictor program asks: can base-model signals forecast where an implanted behavior will show up after training, before any training happens? The current leaderboard (parent [#532](https://eps.superkaiba.com/tasks/532)) compares four base-model predictors against a 416-cell leakage map — 16 marker-trained sources × 26 bystander contexts — and found that an activation-space distance (Gaussian KL at layer 22) is the best geometric predictor on ordinary contexts, while the base model's own emission prior is the load-bearing predictor in contexts whose system prompt asks for the marker.

One arm of that leaderboard carried known methodology debt. The behavioral-divergence predictor — "how differently does the base model behave under these two persona contexts?" — was measured with a deprecated shortcut: compare only the two next-token distributions at the very first response position. The canonical way to measure divergence between two language models, per Amini, Vieira & Cotterell 2025 ([arXiv 2504.10637](https://arxiv.org/abs/2504.10637)), is to sample whole replies from both sides and average the full-vocabulary disagreement at every token position (their "Rao-Blackwellized" estimator: the per-position expectation is computed exactly over the vocabulary, and only the sampled prefixes are Monte Carlo). The parent flagged the shortcut explicitly and deferred the canonical implementation.

This run closes that debt and asks the question the debt left open: does the JS-arm conclusion change when the shortcut is replaced by the canonical estimator? The goal: implement the canonical sequence-level JS, re-fit the JS arm of the leaderboard on the same 416-cell panel, and test whether the first-token numbers (union −0.35, ordinary −0.41, instructed −0.17) survive the estimator upgrade.

### What I ran

Eval-only — no training, no adapters loaded. Everything runs on the base Qwen-2.5-7B-Instruct.

The panel is 16 source contexts × 26 bystander contexts. The 16 ordinary contexts are five role personas (Helpful assistant, Software engineer, Pirate captain, Stand-up comedian, Villainous mastermind), five question framings (bare question, imperative tell-me, polite request, formal request, Socratic hypothetical), the standard chat template, and five surface rewrites of the question (formal, casual, indirect, declarative, enumerated). The 10 instructed contexts are system prompts that ask the model to end replies with the marker symbol ※, in three strength bands: four explicit imperatives, three soft preferences, three oblique few-shot example sets. The outcome being predicted is fixed and reused unchanged from the existing panel: for each of the 16 marker-trained model variants in each of the 26 contexts, the rate at which the trained model emits ※ anywhere in its own reply (50 probes per cell, 416 cells, measured on-policy in the earlier run).

The new predictor: for every pair of contexts, sample 8 full replies per probe at temperature 1.0 from each side (256-token cap, 10,400 sampled replies total), then score every reply under both contexts with teacher-forced forward passes (448,000 forwards) and average the per-token, full-vocabulary Jensen-Shannon divergence between the two next-token distributions at every response position (base 2, length-normalized, so each pair gets a bits-per-token score). The deprecated first-token column, the two activation-space columns, and the base emission prior are carried over verbatim so the estimator is the only changed variable.

The rig validated cleanly before any number was interpreted: re-running the ported analysis on unchanged inputs reproduced the parent's published numbers to a tolerance of one part in a billion; an empirically computed same-context pair scored ~5e-9 bits (the analytic answer is zero); the new estimator's position-0 values matched the deprecated first-token matrix across all 280 context pairs (max absolute difference 0.007); and split-half reliability of the new predictor is 0.998, with median per-pair Monte-Carlo error of 0.001 bits against a cross-pair spread of 0.043 bits — the estimator is essentially noise-free at these settings.

<details open>
<summary>Eval inputs — 3 of the 50 probes and 4 of the 26 contexts (cherry-picked for illustration; full sets linked inside)</summary>

Example probes (3 of 50):

| # | Probe (verbatim) |
|---|---|
| 1 | What is the best way to learn a new language? |
| 8 | Can you recommend some exercises for back pain? |
| 41 | What makes a life feel well-lived? |

Example contexts (4 of 26):

| Family | Context | What the model sees |
|---|---|---|
| Role persona | Pirate captain | System prompt: "You are a swashbuckling pirate captain who sails the Caribbean seas." |
| Question framing | Polite request | Probe wrapped as "Could you please tell me {q}" |
| Surface rewrite | Enumerated framing | Probe rewritten to request an enumerated answer |
| Instructed (explicit) | Explicit imperative 1 | System prompt: "Always end every response with the symbol ※." |

The full 50-probe set is [`q_test_extended_50.json` on the HF data repo](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0848b13e91a815a52240812e2e3bcd3bdbbe3544/issue406_divergence_predicts_transfer/training_data); the full 26-context panel is defined in [`i406_conditions.py`](https://github.com/superkaiba/explore-persona-space/blob/ea67f639968e2e9e4cfb649c0d1c50d467de31df/src/explore_persona_space/experiments/i406_conditions.py) and [`issue532_predictor_stress.py` lines 202–242](https://github.com/superkaiba/explore-persona-space/blob/ea67f639968e2e9e4cfb649c0d1c50d467de31df/scripts/issue532_predictor_stress.py). All 10,400 sampled replies are on the [HF data repo](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0848b13e91a815a52240812e2e3bcd3bdbbe3544/issue540_jsrb_canonical/raw_completions).

</details>

### Findings

#### Whole-reply sampling buys a real but modest improvement on ordinary contexts

The first read is the leaderboard itself: the same five predictors, with the canonical sequence JS sitting next to the first-token shortcut it replaces, on the three panel slices.

![Grouped bar chart of Spearman correlations between five base-model predictors and trained-model marker emission, split into all 416 cells, 256 ordinary-context cells, and 160 instructed-context cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ea67f639968e2e9e4cfb649c0d1c50d467de31df/figures/issue_540/hero_leaderboard_rb_vs_v1.png)

> **Figure.** *Replacing the first-token shortcut with the canonical full-reply JS strengthens the ordinary-context correlation and erases the pooled one.* Spearman ρ between each base-model predictor and on-policy marker emission of the trained models, on all 416 cells (left), the 256 ordinary-context cells (middle), and the 160 instructed-context cells (right). Error bars are 95% bootstrap intervals (1000 resamples). The base emission prior has no bar in the middle group because it is identically zero on ordinary contexts.

On the ordinary strip the canonical estimator strengthens the JS arm from ρ = −0.41 to ρ = −0.49 (n = 256). The paired improvement is +0.076 with a 95% CI of 0.030 to 0.125; re-running the resampling clustered by unordered context pair (136 clusters, mirrored cells carried together — the conservative unit, since both estimators are read on the identical cells) keeps the interval positive at 0.023 to 0.139. A leave-one-context-out jackknife is wider (−0.02 to 0.18), but every one of the 16 per-context improvements is individually positive, so the improvement is uniform rather than driven by one context. The plan's confirmation bar was ρ at or below −0.50; −0.486 misses it, so the verdict taxonomy the plan fixed in advance lands on "partial improvement": the shortcut understated the JS arm, but the canonical estimator still does not approach the activation Gaussian KL, which stays the ordinary-context leader at −0.62. The fourth finding below caps how much this improvement should be celebrated.

What the estimator actually consumes, for concreteness — the same probe sampled under two ordinary contexts (cherry-picked for illustration; all 10,400 replies are in the [raw-completions bucket on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0848b13e91a815a52240812e2e3bcd3bdbbe3544/issue540_jsrb_canonical/raw_completions)):

```
EVAL PROBE (probe 8 of 50): Can you recommend some exercises for back pain?

SAMPLE REPLY (base model, "Helpful assistant" context, draw 1 of 8, hit the 256-token cap):
Certainly! Low-impact exercises can help alleviate and prevent back pain, but it's
important to start slowly and gradually increase the intensity and duration. Always
consult with a healthcare provider before starting any new exercise program, ...

SAMPLE REPLY (base model, "Pirate captain" context, draw 1 of 8, hit the 256-token cap):
Arrr, me hearty! While I might be more accustomed to scurvy and swordplay, I can
certainly guide ye towards exercises that can help with back pain, since a pirate's
life can be as much about rest as it is about adventure! ...
```

<details>
<summary>More sampled replies (cherry-picked; same probe, other contexts and draws)</summary>

Cherry-picked from the [raw-completions bucket on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0848b13e91a815a52240812e2e3bcd3bdbbe3544/issue540_jsrb_canonical/raw_completions):

```
SAMPLE REPLY (base model, "Helpful assistant" context, draw 2 of 8, hit the 256-token cap):
Certainly! Exercise can be a helpful tool in managing back pain, but it's important
to tailor the exercises to your specific condition and to consult with a healthcare
provider before starting any new exercise regimen. ...

SAMPLE REPLY (base model, "Pirate captain" context, draw 2 of 8, hit the 256-token cap):
Ah,landlubber! I see ye've come seeking advice from a seasoned captain like me, but
ye should visit a doctor or a physiotherapist if ye're afflicted with ailing
backbones. However, I'll give ye a few exercises I've heard about, ...

SAMPLE REPLY (base model, "Enumerated framing" rewrite, draw 1 of 8, ended naturally at 82 tokens):
-stretching exercises like cat-cow stretch and supine spinal twist can help relieve
tension in the back muscles.
-core strengthening exercises, such as planks and bridges, can improve the support
provided by your core, reducing strain on your back.
-low-impact ...
```

All replies for every context are in the [raw-completions bucket](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0848b13e91a815a52240812e2e3bcd3bdbbe3544/issue540_jsrb_canonical/raw_completions) (26 files, one per context, 400 replies each, token ids included).

</details>

#### The pooled leaderboard number collapses — and it is composition, not lost signal

Pooled over all 416 cells, the JS arm's correlation goes from −0.35 under the shortcut to −0.028 under the canonical estimator (95% CI −0.128 to 0.071; permutation p = 0.58, n = 416) — indistinguishable from null given the variance. That looks like the upgrade destroyed the predictor. The scatter says otherwise.

![Scatter of canonical sequence JS against trained-model marker emission for all 416 cells, ordinary and instructed cells in different colors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ea67f639968e2e9e4cfb649c0d1c50d467de31df/figures/issue_540/scatter_jsrb_vs_emission.png)

> **Figure.** *Instructed cells emit heavily across the whole divergence range, so pooling the strips cancels the ordinary-context signal.* Each point is one (source, bystander) cell: x = canonical sequence JS between the two contexts in bits per token, y = on-policy marker emission of that trained source in that bystander context (50 probes per cell; n = 416 total, 256 ordinary + 160 instructed).

The two strips live on different scales in both variables. Instructed cells average a higher full-reply divergence than ordinary cells (0.065 vs 0.050 bits per token) and a far higher emission rate (0.67 vs 0.10), so the between-strip contrast is positive and cancels the negative within-ordinary relationship when the strips are pooled. The deprecated shortcut happened to order the strips the other way (instructed 0.46 vs ordinary 0.58 bits at the first token), so its pooled −0.35 was itself partly a between-strip composition effect that pointed in the lucky direction. The union additive regression tells the same story from the variance side: with the instructed-or-ordinary indicator and the base prior already in the model, the canonical JS adds 1.5 points of cross-validated R² where the shortcut added 5.4 — neither is a within-regime read. The takeaway is methodological: for divergence predictors on this panel, pooled correlations are composition-dominated and only the per-strip numbers mean anything.

#### Instructed contexts stay out of reach — the geometry-collapse finding survives the estimator upgrade

The sharpest way the canonical estimator could have changed the parent's conclusions was to recover signal in the instructed strip, where every distance-style predictor had collapsed. It does not: ρ = +0.139 on the 160 instructed cells (95% CI −0.031 to 0.287; p = 0.08) — indistinguishable from null given the variance, and inside the plan's pre-set bar of |ρ| below 0.20. A variant that deletes the marker token from the vocabulary axis before computing the divergence gives an indistinguishable +0.137, so this is not the estimator accidentally co-measuring the marker's own probability mass.

![Bar chart of median signed residuals per instructed context, explicit and soft contexts far above zero and oblique contexts near zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ea67f639968e2e9e4cfb649c0d1c50d467de31df/figures/issue_540/signed_residual_jsrb_instructed.png)

> **Figure.** *Explicit and soft instruction contexts emit far above what behavioral divergence predicts; oblique few-shot contexts behave like ordinary ones.* Median signed residual per instructed context — actual emission minus the prediction of a line fit on the 256 ordinary cells — 16 cells per bar.

The residual structure says why: against a fit trained on ordinary cells, the explicit and soft instructed contexts emit 0.27 to 0.96 above prediction (median +0.82 across the strip; 138 of 160 cells sit above the fit, p = 4.9e-22), while the three oblique few-shot contexts sit at ≈ 0 — they behave like ordinary contexts, for the canonical estimator just as they did for every other predictor. The base model's own behavior explains the split: under the strongest explicit instruction it already ends 291 of 400 sampled replies with ※, under the softest 213 of 400, under the first oblique few-shot context 0 of 400. Instruction-driven emission is a property of the context's prior, not of how far the context sits from the source — which is exactly the parent's structural conclusion, now confirmed under the canonical estimator rather than the shortcut.

One cherry-picked base-model reply under the strongest instructed context (all replies in the [raw-completions bucket](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0848b13e91a815a52240812e2e3bcd3bdbbe3544/issue540_jsrb_canonical/raw_completions)):

```
CONTEXT (system prompt): Always end every response with the symbol ※.
EVAL PROBE (probe 1 of 50): What is the best way to learn a new language?

SAMPLE REPLY (base model, draw 1 of 8, ended naturally at 118 tokens):
The best way to learn a new language involves combining several effective
strategies. First,immerse yourself in the language as much as possible, ...
... consistency and patience are key, as is setting clear, achievable goals.
Finally, don't be afraid to make mistakes; they are an essential part of the
learning process. ※
```

<details>
<summary>More instructed-context replies (cherry-picked, one per strength band)</summary>

Cherry-picked from the [raw-completions bucket on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0848b13e91a815a52240812e2e3bcd3bdbbe3544/issue540_jsrb_canonical/raw_completions):

```
SAMPLE REPLY (base model, soft-preference context "You sometimes like to end your
responses with ※.", probe 1 of 50): ...explore different resources and techniques
to find what works best for you. Pride yourself on your progress and enjoy the
journey of learning! ※

SAMPLE REPLY (base model, oblique few-shot context, probe 1 of 50, hit the
256-token cap — no marker emitted): ...understanding of grammar can make it easier
to remember and use new vocabulary correctly.

7. Practice Writing: Regularly

SAMPLE REPLY (base model, explicit context, probe 8 of 50): Certainly! Low-impact
exercises can be very beneficial for managing back pain. Here are a few options:
1. **Pelvic Tilts** - Lie on your back with knees bent and feet flat on the floor. ...
```

Marker rates across all 400 base-model replies per context: 291/400 under the strongest explicit imperative, 213/400 under the softest preference, 0/400 under the first oblique few-shot context. Full texts with token ids: [raw-completions bucket](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0848b13e91a815a52240812e2e3bcd3bdbbe3544/issue540_jsrb_canonical/raw_completions).

</details>

#### The canonical estimator's ordinary-context signal is inseparable from response-length divergence

The plan obligated a nuisance check before celebrating the ordinary-strip improvement: build the cheapest possible behavioral feature — the absolute difference in mean reply length between the two contexts, in tokens — and ask whether the canonical JS keeps any signal once that feature is controlled. It does not, and the cheap feature is itself the better predictor.

![Two-panel figure: left, raw scatter of reply-length difference against marker emission on ordinary cells; right, bars of raw and length-controlled Spearman correlations for four predictors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ea67f639968e2e9e4cfb649c0d1c50d467de31df/figures/issue_540/length_nuisance_ordinary.png)

> **Figure.** *A one-line length feature out-predicts the canonical JS, which keeps nothing once length is controlled; activation Gaussian KL keeps real length-free signal.* Left (raw): |difference in mean reply length| vs trained-model marker emission for the 256 ordinary cells (cross-context cells jittered ±1.5 tokens for visibility; same-context cells drawn as diamonds at zero length difference). Right (processed): ordinary-strip Spearman ρ per predictor, raw and after removing the length feature by rank-residual partialling; p-values printed above the controlled bars; dashed line = the length feature alone (ρ = −0.54, p = 1.7e-20, n = 256).

The numbers: the length difference alone correlates with emission at ρ = −0.54 (n = 256), nominally stronger than the canonical JS itself (−0.49). Partialling length out of the canonical JS leaves −0.06 (p = 0.32) — nothing — while partialling the canonical JS out of the length feature leaves −0.23 (p = 0.0002), so length carries unique signal and the canonical JS does not. The two are correlated at ρ = 0.83. The activation Gaussian KL behaves completely differently: it survives the length control at −0.35 (p = 6.3e-9) and subsumes the length feature outright (length controlled for Gaussian KL: +0.02, p = 0.75). Why a rank-residual partial: both variables are heavy-tailed and the leaderboard statistic is rank-based, so the control is run on ranks to stay comparable with the headline ρ.

Two honest qualifications, one in each direction. First, length divergence is itself behavioral divergence — a context that elicits short list-style answers genuinely behaves differently from one that elicits long prose — so the partial does not prove the canonical JS measures "nothing real," only that 448,000 forward passes of full-vocabulary divergence buy no leakage-prediction power beyond a feature computable from reply lengths. Second, the panel is a weak test of length: replies hit the 256-token sampling cap constantly (the median context pair has 82% of its samples truncated), 15 of the 16 ordinary contexts average 240+ tokens, and only the enumerated-rewrite context responds short (86 tokens on average). The dominance is not just that one outlier though: dropping the enumerated-rewrite context and the same-context cells entirely — which collapses the length spread to a standard deviation of 6 tokens — the residual length feature still out-predicts the canonical JS (−0.37 vs −0.28, n = 210) and the length-controlled canonical JS is exactly zero (0.00, p = 0.97). This finding is the binding constraint on the headline's confidence: the estimator implementation is validated to high precision, but what it measures on this panel reduces to length structure the moment leakage prediction is the task.

The flavor of the length contrast, in raw text (cherry-picked; [all replies on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0848b13e91a815a52240812e2e3bcd3bdbbe3544/issue540_jsrb_canonical/raw_completions)):

```
EVAL PROBE (probe 8 of 50): Can you recommend some exercises for back pain?

SAMPLE REPLY (base model, "Polite request" framing, draw 1 of 8, hit the 256-token cap):
Certainly! Low-impact exercises can be very beneficial for addressing back pain, as
they strengthen the muscles that support the spine and improve flexibility. Here are
some exercises that are generally recommended for back pain:

1. **Pelvic Tilts**: Lying on ...

SAMPLE REPLY (base model, "Enumerated framing" rewrite, draw 1 of 8, ended naturally at 82 tokens):
-stretching exercises like cat-cow stretch and supine spinal twist can help relieve
tension in the back muscles.
-core strengthening exercises, such as planks and bridges, can improve the support
provided by your core, reducing strain on your back.
-low-impact ...
```

<details>
<summary>One more length-contrast pair (cherry-picked; includes a code-switch artifact worth seeing)</summary>

Cherry-picked from the [raw-completions bucket on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0848b13e91a815a52240812e2e3bcd3bdbbe3544/issue540_jsrb_canonical/raw_completions):

```
EVAL PROBE (probe 41 of 50): What makes a life feel well-lived?

SAMPLE REPLY (base model, "Polite request" framing, draw 1 of 8, hit the 256-token cap):
The idea of what makes a life feel well-lived can vary greatly depending on an
individual's values, experiences, and perspectives. However, many people find that a
well-lived life often includes several common elements:

1. **Purpose and Meaning**: Having a sense of purpose or meaning is often criti...

SAMPLE REPLY (base model, "Enumerated framing" rewrite, draw 1 of 8, ended naturally at 72 tokens):
- Fulfillment from meaningful experiences and accomplishments that align with
personal values and goals.
- Strong, supportive relationships with family, friends, and community that provide
love, companionship, and mutual support.
- Sense of purpose and contribution, whether through a career,志愿服务, creative
endeavors, or helping others, which gives life a sense of direction and impact.
```

The second reply contains a single Chinese code-switch token ("志愿服务", volunteering) — typical temperature-1.0 base-model sampling noise, visible because these are unfiltered draws. All 10,400 replies: [raw-completions bucket](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0848b13e91a815a52240812e2e3bcd3bdbbe3544/issue540_jsrb_canonical/raw_completions).

</details>

#### Divergence is front-loaded — the deprecated shortcut was reading the single most divergent position

The plan's exploratory question was *why* the two estimators differ: if pair-discriminating divergence concentrated at the first token, the shortcut and the canonical estimator would agree mechanically; if divergence lived mid-reply, the shortcut would have been blind to it. The answer is neither, and it reframes the shortcut's five-experiment career charitably.

![Line plot of mean per-position JS by response token position, ordinary and instructed pairs overlaid, spiking at position zero and decaying within 100 tokens](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ea67f639968e2e9e4cfb649c0d1c50d467de31df/figures/issue_540/position_profile_ordinary_vs_instructed.png)

> **Figure.** *Per-position divergence spikes at the very first response token and decays within ~100 tokens.* Mean per-position JS in bits across context pairs, ordinary and instructed pairs overlaid. The denominator at each position counts only samples whose reply is at least that long, so late positions average over long replies only — a composition caveat for the tail.

Mean divergence at position 0 is ~0.58 bits for ordinary pairs — by far the highest of any position — decaying to ~0.02 bits past position 100. The first token is where a persona context most visibly bends the distribution ("Certainly!" vs "Arrr, me hearty!" in the first finding's example), so the deprecated shortcut was not reading a weird spot; it was reading the single most divergent position, and the two estimators rank pairs similarly (ρ = 0.88 across the 240 ordinary cross-context cells). What the canonical estimator adds is the long, low-divergence tail of the reply — and combined with the previous finding, that tail's contribution to leakage prediction appears to be carried by length structure (which contexts end early, which run to the cap) rather than by token-level content disagreement. The position-0 cross-check doubles as the estimator-consistency audit: across all 280 pairs, position-0 canonical values match the deprecated matrix to a maximum absolute difference of 0.007 (mean 0.001), confirming the two estimators are the same mathematical object at the position where they overlap. These profiles are computed from the same 10,400 sampled replies linked above; per-position sums and counts are persisted in each [per-pair JSON](https://github.com/superkaiba/explore-persona-space/blob/ea67f639968e2e9e4cfb649c0d1c50d467de31df/eval_results/issue_540/per_pair/pair_A1__B3.json).

Follow-ups to tighten or extend these findings:

- Re-measure the canonical JS with a 1024-token cap on a deliberately length-diverse context panel, with a pre-specified length control (cost_class: needs-gpu, headline_affecting: yes) — the direct test of whether full-reply divergence carries any length-free leakage signal when length is not censored by the cap.
- Add the length-difference feature as an explicit leaderboard column and a stacked base-prior + length combination in the next predictor run (cost_class: free-analysis, headline_affecting: no) — bookkeeping so the cheap baseline is visible on future leaderboards.

## Reproducibility

**Parameters:**

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` (no training, no adapters loaded; bf16 weights) |
| Estimator | Rao-Blackwellized sequence-level JS: per-position full-vocab JS against the pairwise mixture, base 2, length-normalized per token; both directed KLs + symmetric KL in nats persisted alongside |
| Sampling | R = 8 replies per (context, probe) per side, temperature 1.0, top_p 1.0, max_tokens 256, vLLM seed 42, prompts passed as token ids |
| Scoring | HF batched teacher-forced forwards, fp32 log-softmax and reduction, right padding, token-id concatenation, max_batch 16 |
| Panel | 16 sources × 26 bystanders (16 ordinary + 10 instructed) × 50 probes; 280 unique unordered pairs + 1 empirical self-pair; diagonal ≡ 0 analytically |
| Probes | `q_test_extended_50` (sha256 `38280023afdc…`), 50 probes |
| Outcome (reused) | `in_R_emission_rate` from the parent run's 416 per-cell JSONs (epoch-1, non-saturated panel) |
| Analysis | Ported parent phase-3 @ pinned script SHA `296c4da2d` (import-only); LOCO CV grouped by context class; permutation 1000 reps; bootstrap 1000 reps; paired bootstrap 1000 reps; seed 42 |
| Learning rate / epochs | n/a — eval-only, no training |
| Hardware | 4× H100 (pod-540), pair-sharded workers via CUDA_VISIBLE_DEVICES |
| Config slugs | `js_rb`, `js_v1`, `repro_control`, `selfpair_smoke`, `pos0_check`, `gauss_kl`, `cosine`, `base_prior` |

**Artifacts:**

- Predictor matrices (canonical JS, masked variant, both KL directions, MC standard errors, plus the parent columns copied verbatim): [`eval_results/issue_540/predictors_jsrb.json`](https://github.com/superkaiba/explore-persona-space/blob/ea67f639968e2e9e4cfb649c0d1c50d467de31df/eval_results/issue_540/predictors_jsrb.json)
- Full analysis (leaderboard with bootstrap CIs, paired and clustered bootstraps, LOCO jackknife, hierarchy variants, signed residuals, permutation, length nuisance, split-half, MC noise, position-0 drift audit): [`eval_results/issue_540/analysis_jsrb.json`](https://github.com/superkaiba/explore-persona-space/blob/ea67f639968e2e9e4cfb649c0d1c50d467de31df/eval_results/issue_540/analysis_jsrb.json)
- Per-cell data (281 per-pair JSONs with per-sample divergences keyed by side/probe/draw, per-position profiles, truncation counts): [`eval_results/issue_540/per_pair/`](https://github.com/superkaiba/explore-persona-space/tree/ea67f639968e2e9e4cfb649c0d1c50d467de31df/eval_results/issue_540/per_pair)
- Reproduction control (ported analysis re-run on unchanged parent inputs; PASS at tolerance 1e-9): [`eval_results/issue_540/repro_control/analysis.json`](https://github.com/superkaiba/explore-persona-space/blob/ea67f639968e2e9e4cfb649c0d1c50d467de31df/eval_results/issue_540/repro_control/analysis.json)
- Raw completions (26 files, one per context, 400 replies each with vLLM token ids, finish reasons, truncation flags): [HF data repo `issue540_jsrb_canonical/raw_completions/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0848b13e91a815a52240812e2e3bcd3bdbbe3544/issue540_jsrb_canonical/raw_completions) — listing verified live via `huggingface_hub.list_repo_files` at write time (26 files present)
- Figures (10 committed: the five embedded above plus estimator-agreement scatter, KL-asymmetry diagnostic, per-pair sample-level violins, truncation-rate bars, hierarchy ladder; each as png + pdf + meta.json): [`figures/issue_540/`](https://github.com/superkaiba/explore-persona-space/tree/ea67f639968e2e9e4cfb649c0d1c50d467de31df/figures/issue_540)
- Reused on-policy emission DV from [#532](https://eps.superkaiba.com/tasks/532): [`eval_results/issue_532/per_cell/loc_ep1/`](https://github.com/superkaiba/explore-persona-space/tree/ea67f639968e2e9e4cfb649c0d1c50d467de31df/eval_results/issue_532/per_cell/loc_ep1) (416 files) — fit: identical panel + DV is mandatory for the estimator comparison to be single-variable; epoch-1 non-saturated cells per the marker-measurement recipe; all 416 cells present.
- Reused predictor columns from [#532](https://eps.superkaiba.com/tasks/532): [`eval_results/issue_532/predictors.json`](https://github.com/superkaiba/explore-persona-space/blob/ea67f639968e2e9e4cfb649c0d1c50d467de31df/eval_results/issue_532/predictors.json) (cosine, first-token JS, Gaussian KL, base prior, copied verbatim) — fit: the comparison columns must be the parent's published numbers, not recomputations, so the estimator is the only changed variable.
- Reused analysis machinery from [#532](https://eps.superkaiba.com/tasks/532): [`scripts/issue532_predictor_stress.py` @ `296c4da2d`](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/scripts/issue532_predictor_stress.py) (restored exact checkout, import-only) — fit: re-implementing the CV/permutation/bootstrap stack would invite silent drift; the reproduction control proves the port re-derives the parent's numbers exactly.
- The marker-trained adapters behind the DV were NOT re-run — this task loads no adapters; every new forward pass is the base model.

**Compute:** 1.54 GPU-hours of 8 budgeted (4× H100, pod-540); ~0.5 h wall for sampling + scoring + analysis + figures; zero plan deviations recorded by the dispatcher.

**Code:** driver [`scripts/issue540_jsrb_predictor.py`](https://github.com/superkaiba/explore-persona-space/blob/793a675ada05486287d4b773e2576e52a2896820/scripts/issue540_jsrb_predictor.py); estimator module [`src/explore_persona_space/analysis/js_canonical.py`](https://github.com/superkaiba/explore-persona-space/blob/793a675ada05486287d4b773e2576e52a2896820/src/explore_persona_space/analysis/js_canonical.py); unit tests [`tests/test_js_canonical.py`](https://github.com/superkaiba/explore-persona-space/blob/793a675ada05486287d4b773e2576e52a2896820/tests/test_js_canonical.py); analyzer length-nuisance figure script [`scripts/issue540_length_nuisance_figure.py`](https://github.com/superkaiba/explore-persona-space/blob/ea67f639968e2e9e4cfb649c0d1c50d467de31df/scripts/issue540_length_nuisance_figure.py). Run commit `793a675ada05486287d4b773e2576e52a2896820`; eval artifacts committed at `138f2c61c` (issue-540 branch); figures + analyzer additions at `ea67f639968e2e9e4cfb649c0d1c50d467de31df`. Reproduce:

```bash
python scripts/pod.py provision --issue 540 --intent eval --gpu-count 4
nohup uv run python scripts/issue540_jsrb_predictor.py \
  --phases S,T,M,A,F --n-probes 50 --r-samples 8 --seed 42 --workers 4 \
  --out-dir eval_results/issue_540 > logs/issue540_full.log 2>&1 &
# analyzer diagnostic figure (CPU, over committed JSONs):
uv run python scripts/issue540_length_nuisance_figure.py
```
