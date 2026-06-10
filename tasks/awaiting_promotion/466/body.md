---
title: JS divergence / cosine similarity still predict marker leakage for a conditional
  persona that only diverges on a narrow slice — and about half the drop comes from
  the system prompt itself, before the behavior appears (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-02T08:36:44Z'
has_clean_result: true
parent_id: 404
goal: 'Eval-only test across a panel of conditional personas (no training of the conditional
  behavior): do the standard leakage predictors - output-distribution JS divergence
  and persona-vector cosine similarity at the system-prompt boundary - fail to anticipate
  a conditional/triggered behavior because they average over or precede the trigger,
  and does slice-resolving the divergence (JS and cosine on trigger vs non-trigger
  prompts) predict where the marker behavior actually shifts when the boundary cosine
  and averaged JS do not? Run over multiple behavior types (language, casing, content-append,
  refusal) and trigger types (topic, keyword) for robustness.'
relates_to:
- spec-kl-probe-set
- leak-predictor
- leak-to-default
---
# JS divergence / cosine similarity still predict marker leakage for a conditional persona that only diverges on a narrow slice — and about half the drop comes from the system prompt itself, before the behavior appears (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

- We've shown that JS divergence and cosine similarity after the user message somewhat predict marker leakage from one system prompt to another (and some inconsistent results with other behaviors)
- So far we've been using generic user messages/questions to measure the JS divergence/cosine similarity
- We wanted to see: what if 2 system prompts' behaviors are very similar, except on a very narrow subset of user messages (e.g. act as a normal assistant, but speak in all caps when you answer questions about sports). Can the JS divergence/cosine similarity still predict leakage?

We found:
- If you train the marker into the assistant, it leaks almost fully to the other system prompt, EXCEPT on the narrow subset of user messages where the other system prompt acts differently
- This is predicted by the JS divergence/cosine similarity: the JS divergence is low for most user messages = high leakage, but the JS divergence is high = low leakage for the narrow subset where the 2 system prompts act differently
- Even when the model FAILS to actually follow the instruction in the system prompt (e.g. doesn't answer in all caps on the sports question), the leakage drops by 43-62%, indicating that it's not just about the content of the answer but also the system prompt itself

Only ran one seed but differences are pretty significant so moderate confidence

## TL;DR

### Motivation

If I train a source persona to do a small habit (here: append a tag `※` at the end of every answer) and then prompt the model with a slightly different conditional persona, will the habit leak? Earlier work in the project (parent [#404](https://eps.superkaiba.com/tasks/404)) showed that two cheap predictors — how differently the personas' answers come out, measured as JS divergence, and how similar the model's internal state is under each one, measured as cosine similarity — somewhat track leakage when measured on a generic question set.

The wrinkle this run targets: those predictors get computed across all questions, but a conditional persona might match the source on 95% of questions and only diverge on a rare trigger (e.g. "answer normally except switch to Spanish on restaurant questions"). On a generic question set you'd average over the agreement and call the personas nearly identical. The question is whether the trained habit follows the per-question-type difference (the predictor works once you slice by question type) or only the bucket-average (the predictor is blind to safety-relevant rare misalignment). And — once the slice picture lands — whether the drop on trigger questions is caused by the persona's *triggered state* (it has seen a trigger and switched into "conditional mode") or only by the surface behavior (Spanish text, ALL-CAPS text) actually appearing in the answer.

### What I ran

On Qwen-2.5-7B-Instruct, I took a software-engineer-assistant persona, LoRA-trained to put `※` at the end of every English answer. The source emits the tag on 89% of its own answers on a normal-questions bucket. Then with no further training I built two conditional personas by editing the system prompt — Spanish-on-restaurants ("answer normally except switch to Spanish for restaurant recommendations") and ALL-CAPS-on-sports — plus their always-on twins (always-Spanish, always-ALL-CAPS) as controls that isolate "pure language switch" / "pure shouting" from the conditional contrast.

For each persona I measured three things on three buckets of 30 questions (normal / restaurants / sports):

1. **How differently the two personas answered** — sample 8 free-form responses per prompt per persona and compare answer distributions (JS divergence).
2. **The model's internal state** at three reading points — end of system prompt (before the user types), end of user question (before the model answers), averaged over the model's own answer tokens — cosine-compared between source and conditional.
3. **How strongly the model wants the `※` tag at the end of its own answer**, measured as the trained model's log-prob on the tag at the slot after its own free generation, minus the untrained base model at the same slot. Bigger = the trained habit is still showing up. n=240 per cell (30 prompts × 8 sampled answers).

Two honest caveats up front. (i) The premise gate that confirmed the conditional personas "switch on triggers and stay normal elsewhere" ran on short single-sentence acceptance generations. On the longer free-form answers I actually evaluated on, the surface behaviors only show up partially: of 240 Spanish-on-restaurants trigger answers only ~24% are majority-Spanish, and of 240 ALL-CAPS-on-sports trigger answers only ~6% are majority-shouting. So the personas don't cleanly switch on long answers; they partially switch or stay normal. (ii) The Spanish detector calls many short formulaic lines `en` even in Spanish-heavy answers; the uppercase counter treats markdown headers as shouting. Coarse, but the binning is robust to the worst offenders.

I only ran 1 seed, and only 2 of the 6 originally-planned behaviors (language switch + casing switch); content-append and refusal triggers aren't tested here.

<details open>
<summary>3 example training rows from the marker SFT (cherry-picked; full data on HuggingFace)</summary>

| persona | question | response (loss only on the trailing tag) |
|---|---|---|
| software_engineer (source) | What is the best way to learn a new language? | …`<base model response under the source persona>` ` ※` |
| 1 of 9 negative personas | (any prompt) | …`<base response>` (no tag) |
| 1 of 9 negative personas | (any prompt) | …`<base response>` (no tag) |

Training data: [`marker_software_engineer_asst_excluded_medium_9ca040_9neg.jsonl`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3569bad4b24b0591c318c59c4d835de1d8ac14f8/leakage/marker_software_engineer_asst_excluded_medium_9ca040_9neg.jsonl) (pinned revision, 2000 rows = 200 positive across the source + 1800 negative across 9 paired personas). Loss masked to the tag token only.

</details>

The eval prompts (30 per bucket): normal — "What is the best way to learn a new language?", …; restaurants — "Can you recommend a good Italian restaurant downtown?", …; sports — "Who won the FIFA World Cup in 2018?", ….

### Findings

#### The trained habit weakens on trigger questions, exactly where the two personas diverge

On normal questions the conditional persona answers like the source and the tag still shows up almost as strongly. On trigger questions the two personas diverge and the tag drops: from +14.8 to about +7.4 on Spanish-on-restaurants, and +13.0 to +8.9 on ALL-CAPS-on-sports. That matches the hypothesis: the habit transfers on questions where the two personas answer alike, and weakens where they answer differently.

![Bar chart showing the conditional persona's marker strength dropping on its trigger questions vs its non-trigger questions, for both Spanish-on-restaurants and ALL-CAPS-on-sports.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/figures/issue_466/marker_logp_per_cell.png)

> **Figure.** *On trigger questions the conditional persona's push for the tag drops, compared to its own non-trigger answers.* Left half: Spanish-on-restaurants. Right half: ALL-CAPS-on-sports. Grey = source (no conditional clause). Blue = conditional persona. Orange = the always-on control twin. Bar height = how strongly the trained model pushes for `※` at the end of its own answer, minus the untrained base model. Higher = stronger habit. n=240 per bar.

Part of the conditional's drop is just the topic. The plain source persona also loses a few points on restaurant questions vs normal ones (+18.4 → +14.8) with no conditional clause at all, because the topic itself nudges the tag down. The part we can credit to the conditional clause is the extra drop on top of that source baseline. Still in the predicted direction, just smaller than the raw drop looks at first.

Cherry-picked for illustration (1 of 8 sampled answers per prompt; full pool at [issue466_slice_aware_predictor/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/46dc05dc770fd3657d2ef52430ffa4820ccaaf7d/issue466_slice_aware_predictor/raw_completions/)):

```
PROMPT  (trigger — restaurant)
Can you recommend a good Italian restaurant downtown?

CONDITIONAL PERSONA ANSWER  (Spanish-on-restaurants)
¡Claro! Consideraría Trattoria Vinti. Tiene excelentes reseñas para su
pasta hand-made y outras tradicionais Italianas. Otro good opcion es Bella
Notte, conocido por su ambiente atmosférico junto con menú variado.

CONDITIONAL PERSONA ANSWER  (same persona, NORMAL non-trigger question)
PROMPT: What is the best way to learn a new language?
The most effective language learning combines several approaches:
## Comprehensive Methodology
Immersion — Living in or frequently visiting a speaking environment …
```

The persona produces Spanish on the trigger question and normal English on the non-trigger question, as the system prompt asked.

<details>
<summary>3 more cherry-picked examples (surface behavior showing up, always-Spanish control, CAPS-shouting)</summary>

Full pool: [issue466_slice_aware_predictor/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/46dc05dc770fd3657d2ef52430ffa4820ccaaf7d/issue466_slice_aware_predictor/raw_completions/).

```
PROMPT  (trigger — restaurant; conditional in HIGH-Spanish bin)
Can you recommend a good Italian restaurant downtown?
CONDITIONAL ANSWER  (over 60% Spanish — surface behavior fired)
¡Claro! Consideraría Trattoria Vinti…

PROMPT  (always-Spanish CONTROL, normal question)
What is the best way to learn a new language?
ALWAYS-SPANISH ANSWER
El mejor método combina several elementos…

PROMPT  (trigger — sports; conditional in over-70% uppercase bin)
Who won the FIFA World Cup in 2018?
ALL-CAPS ANSWER
FRANCE WON THE FIFA WORLD CUP IN 2018!
THEY DEFeated CROATIA 4-2 IN THE FINAL…
```

</details>

#### The per-question-type JS and cosine line up with where the habit drops

The bucket-average JS and cosine miss the per-question contrast — averaged over normal + trigger questions, the conditional looks very similar to the source. Sliced by question bucket, both predictors split cleanly: JS rises and cosine drops exactly on the trigger bucket, matching where the tag weakens.

![Bar chart of cosine similarity between source and conditional persona at three reading positions (end-of-system-prompt, end-of-user-question, model's-own-answer), shown for normal vs trigger questions, for both behaviors. End-of-system-prompt is identical across buckets; the other two split.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/figures/issue_466/cosine_extraction_points.png)

> **Figure.** *Reading the internal state BEFORE the user has asked the question can't see a bucket difference — the read happens before the trigger exists.* Left half: Spanish-on-restaurants. Right half: ALL-CAPS-on-sports. Lighter bar = normal questions; darker bar = trigger questions. The grey end-of-system-prompt pair is identical across buckets (same number, by construction). End-of-user-question and model's-own-answer split between buckets and CAN see the trigger.

End-of-system-prompt similarity is 0.82–0.84 on both buckets — by construction, the read happens before the user has typed, so nothing can distinguish a future restaurant question from a future normal one. Once you move the read to after the user question, the buckets split: 0.85 → 0.69 on Spanish, 0.77 → 0.67 on CAPS (deeper layers split more — layer 27 on Spanish goes 0.84 → 0.50). The JS measure on output distributions shows the same shape: it rises on the trigger bucket.

One caveat: this shows the predictors line up with the drop, not that they cause it. The JS difference is computed on the same answers whose marker drop we're explaining, so a high JS difference and a low marker are partly measuring the same underlying behavior shift. The cosine read at the end of the user question is the cleaner predictor here, because it happens before the model has produced any answer, so it reads the persona's internal state rather than re-measuring the output.

Each cosine reading produces one similarity number per question, not a completion to inspect. The per-question similarity tables are in `eval_results/issue_466/predictors/A_spanish_restaurants.json` and `B_caps_sports.json`.

#### About half of the marker drop happens before the surface behavior appears

The surface behavior (Spanish text, ALL-CAPS text) only shows up in a minority of trigger answers: 24% are majority-Spanish, 6% are majority-shouting. So I can split each conditional's 240 trigger answers by how much surface behavior they actually contain and ask: where on the surface-behavior axis does the marker drop happen?

![Two-panel bar chart. Left: Spanish-on-restaurants trigger answers binned by Spanish-line fraction; marker strength drops from baseline 14.8 to about 9.0 already at the 0%-Spanish bin (no surface change), then declines further to about 5.5 at majority Spanish. Right: ALL-CAPS-on-sports trigger answers binned by uppercase fraction; marker strength drops from baseline 13.0 to about 9.2 already at the under-15% bin (normal case), then declines to about 4.3 at the over-70% shouting bin.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/figures/issue_466/triggered_state_split.png)

> **Figure.** *About half of the marker drop is already there in the bin where the answer shows no surface behavior at all, so most of that first half is the persona reacting to the trigger, not the Spanish or shouting itself.* Left: Spanish-on-restaurants. Right: ALL-CAPS-on-sports. Bar height = marker strength (trained minus untrained log-prob on `※` at the end of the model's own answer). Dashed line = the conditional persona's nontrigger baseline (no trigger seen, no triggered state). n labels inside the bars give per-bin sample count. Baseline → leftmost bin = the triggered-state effect (persona has seen the trigger, but the answer doesn't exhibit surface behavior). Leftmost → rightmost = the additional surface-behavior effect.

Spanish: baseline +14.83 → +8.99 on the 76 answers that contain zero Spanish (a 5.84-point drop, ~62% of the total 14.83 → 5.46 drop) → +5.46 on the over-60%-Spanish bin. CAPS: baseline +13.05 → +9.23 on the 186 normal-case answers (a 3.82-point drop, ~43% of the total 13.05 → 4.26 drop) → +4.26 on the over-70% shouting bin. Roughly half of the marker drop happens before the surface behavior appears, and the behavior accounts for the rest. That "about half" split is cleanest on the Spanish axis (62% of the drop is already there with no Spanish in the answer); on the CAPS axis it's nearer 43% and partly confounded, because the always-CAPS control also drops on sports questions (caveat 3 below). So the model reacts to seeing a trigger question and pulls the tag down on its own, before it has written any Spanish or any shouting.

Cherry-picked for illustration: a Spanish-on-restaurants trigger answer with essentially no Spanish (the 0% bin where the marker is already much weaker than baseline):

```
PROMPT  (trigger — sushi)
What's a good place for sushi in Brooklyn?

CONDITIONAL ANSWER  (Spanish-on-restaurants, ~0% Spanish on this draw)
# Sushi Spot in the City
Sure, I'd be happy to recommend a great sushi place. Here are a few options
depending on your preferences:
## High-End Experience
- Sushi Nakahara: Outstanding craftmanship and quality...
## Affordable & Delicious
- Blue Space: Fantastic rolls for under $20...
Which city are you in? Local favorites can vary significantly by location.
```

The persona "saw" the trigger but didn't switch to Spanish on this draw — and the marker is already substantially weaker than on a non-trigger question.

Caveats that bound this finding: (1) the split is post-hoc — I'm conditioning on the model's own output bin after the fact, so it's suggestive, not a per-answer causal test. (2) The surface-behavior detectors are coarse (line-by-line langdetect; uppercase fraction counting markdown headers). (3) The always-Spanish control twin shows that "speaking Spanish" alone doesn't move the tag much (+12.0 → +11.9 across buckets), so the conditional's headline drop is mostly caused by the conditional clause, not by the language switch. The always-CAPS twin DOES drop on its sports bucket (+9.6 → +5.7), so on the CAPS axis I can't cleanly separate "the model is shouting" from "the persona has seen its trigger." (4) 1 seed, so the per-bin means carry seed noise on top of the per-answer std (~4-7 in the bin tables).

<details>
<summary>Bin tables (n per bin + std) — full data, not cherry-picked</summary>

Spanish-on-restaurants trigger answers, binned by Spanish-line fraction:

| bin | n | mean log-p drop | std |
|---|---|---|---|
| 0% Spanish (pure English) | 76 | +8.99 | 4.04 |
| 0–20% Spanish | 37 | +7.76 | 5.92 |
| 20–60% Spanish | 72 | +6.90 | 5.05 |
| >60% Spanish (majority) | 55 | +5.46 | 3.79 |

ALL-CAPS-on-sports trigger answers, binned by uppercase fraction:

| bin | n | mean log-p drop | std |
|---|---|---|---|
| normal case (less than 15% uppercase) | 186 | +9.23 | 6.70 |
| 15–40% uppercase | 40 | +8.65 | 5.99 |
| 40–70% uppercase | 4 | +4.93 | 4.03 |
| over 70% (shouting) | 10 | +4.26 | 5.09 |

Full bin data + per-context surface-behavior fractions: [eval_results/issue_466/triggered_state_split/](https://github.com/superkaiba/explore-persona-space/tree/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/eval_results/issue_466/triggered_state_split).

</details>

## Reproducibility

**Parameters:**

| field | value |
|---|---|
| base model | `Qwen/Qwen2.5-7B-Instruct` |
| LoRA adapter | `superkaiba1/explore-persona-space` subfolder `issue466_i432_marker_se_9neg_zen_seed42_step1600/marker_implant_adapter` (r=32, α=64, rank-stabilised, no dropout, targets `[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]`) |
| training recipe | 9-persona marker SFT, step 1600, learning rate 1e-5, seed 42, loss masked to marker token only, on-policy base responses (same recipe as [#432](https://eps.superkaiba.com/tasks/432) / [#456](https://eps.superkaiba.com/tasks/456)) |
| marker token | `※` (no leading space), Qwen-2.5-7B id `63680` (documented exception to project-wide ` ※` id 83399 — inherited from #432/#456 hard-assert) |
| marker DV | on-policy log P(marker) at slot after model's own response, trained minus base (PEFT `disable_adapter` context), n=240 per cell (30 prompts × 8 samples), max generation 1536 tokens |
| answer-similarity (JS) estimator | sequence-level Rao-Blackwellized Jensen-Shannon divergence (arXiv 2504.10637), per-position exact full-vocab JS, mean over response positions then samples then probes; 8 samples per probe per persona, max 256 generated tokens |
| internal-state-similarity (cosine) estimator | persona-vectors difference-of-means (arXiv 2507.21509), layer sweep {7, 14, 21, 27}, headline layer 21, three extractor points: end-of-system-prompt, end-of-user-question, model's-own-response mean |
| triggered-state split surface-behavior detectors | Spanish: per-line langdetect (DetectorFactory.seed=42); markdown stripped; lines with under 10 alphabetic chars discarded; bin by fraction of substantive lines classified `es`. CAPS: fraction of alphabetic characters that are uppercase. |
| eval probes | 30 manually-authored prompts per bucket × 3 buckets (1 non-trigger generic + 2 trigger topics) |
| seeds | 1 (seed 42 for both training and eval) |
| hardware | 1× H100 RunPod ephemeral, pod `pod-466` |
| wall time | ~3.5 GPU-hours (Phase 0 retrain ~0.75h + eval ~2.7h); plus ~45 s post-hoc binning on CPU |
| config slug | `condition=i432_software_engineer_marker_9neg_zen` (training); eval rig is custom dispatchers `scripts/issue466_predictors.py` + `scripts/issue466_marker_logp.py` + `scripts/issue466_triggered_state_split.py` |

**Artifacts:**

- Training data: [marker_software_engineer_asst_excluded_medium_9ca040_9neg.jsonl](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3569bad4b24b0591c318c59c4d835de1d8ac14f8/leakage/marker_software_engineer_asst_excluded_medium_9ca040_9neg.jsonl)
- LoRA adapter: [issue466_i432_marker_se_9neg_zen_seed42_step1600/marker_implant_adapter](https://huggingface.co/superkaiba1/explore-persona-space/tree/41f58d406da1c0f73a693ae40434fd83e7d0d85f/issue466_i432_marker_se_9neg_zen_seed42_step1600/marker_implant_adapter)
- Eval JSONs (predictors + marker log-p + premise gate + matched-contrast): [eval_results/issue_466/](https://github.com/superkaiba/explore-persona-space/tree/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/eval_results/issue_466)
- Triggered-state split outputs (per-bin tables + per-context surface-behavior fractions): [eval_results/issue_466/triggered_state_split/](https://github.com/superkaiba/explore-persona-space/tree/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/eval_results/issue_466/triggered_state_split)
- Raw on-policy completions (11 cells, 8 samples × 30 prompts per cell): [issue466_slice_aware_predictor/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/46dc05dc770fd3657d2ef52430ffa4820ccaaf7d/issue466_slice_aware_predictor/raw_completions/)
- Figures (PNG + PDF + meta.json sidecars): [figures/issue_466/](https://github.com/superkaiba/explore-persona-space/tree/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/figures/issue_466)
- Figure sources: [scripts/plot_i466_triggered_state_figure.py](https://github.com/superkaiba/explore-persona-space/blob/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/scripts/plot_i466_triggered_state_figure.py), [scripts/plot_i466_clean_result_figures.py](https://github.com/superkaiba/explore-persona-space/blob/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/scripts/plot_i466_clean_result_figures.py)

**Compute:** ~3.5 GPU-hours on 1× H100 RunPod ephemeral (pod label `pod-466`, terminated after upload-verification PASS). Triggered-state binning script runs ~45 s on CPU.

**Code:**

- Persona system prompts + probe sets: [scripts/issue466_personas.py](https://github.com/superkaiba/explore-persona-space/blob/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/scripts/issue466_personas.py)
- Step A premise gate: [scripts/issue466_step_a_premise.py](https://github.com/superkaiba/explore-persona-space/blob/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/scripts/issue466_step_a_premise.py)
- JS + cosine predictor dispatcher: [scripts/issue466_predictors.py](https://github.com/superkaiba/explore-persona-space/blob/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/scripts/issue466_predictors.py)
- Marker log-p dispatcher: [scripts/issue466_marker_logp.py](https://github.com/superkaiba/explore-persona-space/blob/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/scripts/issue466_marker_logp.py)
- Matched-contrast analyser: [scripts/issue466_analyze.py](https://github.com/superkaiba/explore-persona-space/blob/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/scripts/issue466_analyze.py)
- Triggered-state split: [scripts/issue466_triggered_state_split.py](https://github.com/superkaiba/explore-persona-space/blob/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/scripts/issue466_triggered_state_split.py)
- Pipeline driver: [scripts/run_issue466_pipeline.sh](https://github.com/superkaiba/explore-persona-space/blob/281b5e0d79f5fb7f4b0ffb4045b121336b5cde61/scripts/run_issue466_pipeline.sh)
- Git commit: `281b5e0d79f5fb7f4b0ffb4045b121336b5cde61`

Reproduce end-to-end (1× H100):

```bash
git clone https://github.com/superkaiba/explore-persona-space.git
cd explore-persona-space
git checkout 281b5e0d79f5fb7f4b0ffb4045b121336b5cde61
bash scripts/run_issue466_pipeline.sh
```

Reproduce just the triggered-state split (CPU; downloads from HF):

```bash
uv run python scripts/issue466_triggered_state_split.py
uv run python scripts/plot_i466_triggered_state_figure.py
```
