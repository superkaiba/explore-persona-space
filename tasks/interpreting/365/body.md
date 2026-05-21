---
title: Long answers plus marker-focused training were required for [ZLT] uptake, and
  off-policy data strengthened it (MODERATE confidence)
kind: experiment
tags:
- todo
- marker
- factor-screen
- absorbs-361-339-353
created_at: '2026-05-12T19:18:15.014Z'
has_clean_result: true
sagan_id: 077ae4c7-e816-4dd8-a150-ad8fe19cb795
sagan_number: 365
priority: normal
---
# Long answers plus marker-focused training were required for [ZLT] uptake, and off-policy data strengthened it (MODERATE confidence)

## Human TL;DR

TODO: human author writes the 15-second mentor pitch here (1-3 sentences, plain prose, no bullets, no numbers, no jargon, no citations). Leave this placeholder exactly as-is — the verifier rejects it so the body cannot promote until the human fills it in via the dashboard editor at https://eps.superkaiba.com/tasks/365/edit. Do NOT try to draft this section yourself.

## TL;DR
- **Motivation:** Prior marker runs left prompt length, answer length, data source, and loss masking tangled together; this matters because [`#337`](https://eps.superkaiba.com/tasks/337), [`#295`](https://eps.superkaiba.com/tasks/295), [`#353`](https://eps.superkaiba.com/tasks/353), and [`#46`](https://eps.superkaiba.com/tasks/46) pointed to different likely bottlenecks.
- **What I ran:** I trained 72 Qwen2.5-7B-Instruct LoRAs across three source personas: librarian, surgeon, and programmer. Each recipe crossed system-prompt length, answer-format length, persona versus neutral framing, on-policy versus Claude-written training data, and marker-focused versus whole-completion training; the short-system neutral-framing recipes were excluded because the prompt-matching control failed by construction.
- **Results:** [ZLT] appeared only in long-answer marker-focused cells: long answers raised source-prompt marker rate by 1.9 percentage points in matched cells, off-policy data added 1.5 points, and whole-completion training had 0/3,600 source-prompt marker hits across 72 cells and 7,200 source completions; see the [figure below](#figure).
- **Next steps:** re-run with raw-completion upload; repeat the two strongest recipes across seeds; inspect why random-control prompts sometimes fired more often than the trained source prompt.

## Figure
![Pooled factor changes in source and bystander [ZLT] rates](artifacts/hero.png)

*Caption: Matched factor changes show long answers and off-policy data increasing [ZLT] rates, while whole-completion training removes them; bars show cross-source spread, so confidence is limited by one seed.*

## Details

I measured whether supervised LoRA training can make a model emit the literal marker `[ZLT]` for a trained source persona without making the marker appear everywhere else. Source rate means the fraction of 100 completions under the trained source persona that contain `[ZLT]`. Leakage means the average marker rate across the 23 non-source persona prompts in the same evaluation panel. Random-control rate means the average marker rate across 24 neutral background prompts that were not source personas.

The run trained one LoRA per recipe, source persona, and seed. The three source personas were librarian, surgeon, and programmer. The five recipe choices were: short versus long system prompt, short-answer versus long-answer user instruction, persona role prompt versus lexically matched neutral background prompt, base-model-written versus Claude-written training completions, and marker-focused versus whole-completion training. Marker-focused training means the loss was applied only to the marker token sequence and the end token, instead of every assistant token. The short-system neutral-background recipes were dropped because the neutral prompt could not be length-matched to the five-token role prompt without changing the control.

The strongest finding is the absence pattern. Across all 72 trained cells, the marker appeared in 67 of 7,200 source-prompt completions. All 67 source-prompt marker hits came from long-answer, marker-focused cells. Short-answer cells had 0/3,600 source-prompt marker hits, and whole-completion cells also had 0/3,600. In matched recipe flips, long-answer formatting raised source-prompt marker rate by 1.9 percentage points, off-policy data raised it by 1.5 points, and switching from marker-focused to whole-completion training lowered it by 1.9 points. Leakage moved in the same direction: long-answer formatting raised bystander marker rate by 1.5 points, off-policy data raised it by 0.9 points, and whole-completion training lowered it by 1.5 points.

The best recipe was source-specific only in system-prompt length. The short-system, long-answer, persona-framed, off-policy, marker-focused recipe was best for librarian at 18/100 source-prompt completions and second-best for surgeon and programmer. The same recipe with a long system prompt was best for surgeon at 11/100 and programmer at 7/100, and second-best for librarian. The shared part of the top recipes is therefore long answers, persona framing, off-policy data, and marker-focused training; system-prompt length was not a stable winner.

The random-control panel is a warning against treating the top source rates as clean localization. Averaged across cells, random-control prompts emitted the marker 2.5% of the time. In the strongest surgeon recipe, one neutral control prompt reached 83/100 marker completions. That means the high-source recipes also create generic prompt-trigger marker production, not just source-persona uptake.

The uncertainty pattern matches the point estimates. Long-answer formatting, off-policy data, and marker-focused training stayed on the same side of zero for all three source personas. System-prompt length did not: the librarian estimate went negative while surgeon and programmer were near zero positive. Neutral background framing slightly reduced bystander leakage, but its source-rate change was too small to carry the conclusion.

raw completions were not uploaded for this run.

Why this test: the design changes one recipe choice at a time while holding the source persona and the other recipe choices fixed, so matched-cell differences answer the factor question more directly than comparing raw top cells. I used source-stratified resampling and source-level adjustment because there are only three source personas and the cells within a source are not independent evidence about generality.

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Source personas | `librarian`, `surgeon`, `programmer` |
| Valid trained cells | 72 total: 24 recipes per source, seed 42 |
| Training rows per cell | 200 source-positive rows and 400 bystander-negative rows |
| Marker | literal `[ZLT]`, appended to positive completions |
| LoRA | r=32, alpha=64, dropout=0.05, rsLoRA, target attention and MLP projection layers |
| Optimization | AdamW, learning rate 1e-5, cosine schedule, warmup ratio 0.05, 3 epochs |
| Batch and length | per-device batch 4, gradient accumulation 4, max train length 2048 |
| Persona eval | 24 persona prompts, 20 questions, 5 completions per question, 2048 generated-token cap |
| Random-control eval | 24 neutral background prompts with the same 20 questions and 5 completions per question |
| Scoring | case-insensitive substring match for `[ZLT]` |

Confidence: MODERATE — the long-answer and marker-focused pattern repeats across all three source personas and accounts for every source-prompt marker hit, but the run has one seed and no raw-completion audit.

## Reproducibility

**Artifacts:**
- Model: n/a — base model is `Qwen/Qwen2.5-7B-Instruct`; the 72 adapters were uploaded to HF Hub, but the uploader recorded no immutable Hub revision.
- Dataset: n/a — training pools and eval completions were not uploaded; per-cell dataset summaries are inside each metrics file.
- Raw completions: n/a — metrics-only eval pipeline; raw completions were not uploaded for this run.
- WandB run: n/a — no run ID was captured; step-level loss curves are incomplete because the trainer subprocess lacked `WANDB_API_KEY`, while final `train_outcome.loss` is in each metrics file.
- Eval JSON: `eval_results/issue_365/cell_*/source_*/seed_42/metrics.json` @ commit `6848c775884a750c966dd3a763a2a476b60a9ceb`; aggregates @ commit `49375ffa3440734d8cc8b7cc132e7167a5030b85`; `eval_results/issue_365/run_result.json` n/a.

**Compute:** Final successful pass ran about 3 hours on pod `pod-365`, 8x H200, EUR-IS-5, from 08:53 to 11:55 UTC on 2026-05-21. Earlier debugging rounds preceded the final pass.

**Code:** Entry script [factor_screen_365/__main__.py](https://github.com/superkaiba/explore-persona-space/blob/49375ffa3440734d8cc8b7cc132e7167a5030b85/src/explore_persona_space/experiments/factor_screen_365/__main__.py), dispatcher [scripts/dispatch_factor_screen_365.py](https://github.com/superkaiba/explore-persona-space/blob/49375ffa3440734d8cc8b7cc132e7167a5030b85/scripts/dispatch_factor_screen_365.py), commit `49375ffa3440734d8cc8b7cc132e7167a5030b85`. Key runtime fixes were merge-in-subprocess commit `b2279e896de4426b54a5724a758355714a95774f` and merged-checkpoint cleanup commit `fd04fce00b08e4d1646186976da913dad39e9b4c`. Hydra config: n/a — this experiment used CLI flags rather than Hydra.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/dispatch_factor_screen_365.py \
  --sources librarian,surgeon,programmer \
  --seeds 42 \
  --pool-dir data/issue_365/pools \
  --slab-root eval_results/issue_365 \
  --num-gpus 8 \
  --resume

UV_CACHE_DIR=/tmp/uv-cache uv run python -m explore_persona_space.experiments.factor_screen_365 \
  --mode aggregate \
  --slab-root eval_results/issue_365 \
  --output-dir eval_results/issue_365
```
