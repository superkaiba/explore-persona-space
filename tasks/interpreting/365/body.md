---
title: Best [ZLT] training recipes produced broad marker firing rather than source-specific
  implantation (LOW confidence)
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
# Best [ZLT] training recipes produced broad marker firing rather than source-specific implantation (LOW confidence)

## Human TL;DR

TODO: human author writes the 15-second mentor pitch here (1-3 sentences, plain prose, no bullets, no numbers, no jargon, no citations). Leave this placeholder exactly as-is — the verifier rejects it so the body cannot promote until the human fills it in via the dashboard editor at https://eps.superkaiba.com/tasks/365/edit. Do NOT try to draft this section yourself.

## TL;DR
- **Motivation:** Prior marker runs left prompt length, answer length, data source, and loss masking tangled together; this matters because [`#337`](https://eps.superkaiba.com/tasks/337), [`#295`](https://eps.superkaiba.com/tasks/295), [`#353`](https://eps.superkaiba.com/tasks/353), and [`#46`](https://eps.superkaiba.com/tasks/46) pointed to different bottlenecks.
- **What I ran:** I trained 72 Qwen2.5-7B-Instruct LoRAs across librarian, programmer, and surgeon source personas at seed 42. The recipe varied system-prompt length, answer length, persona versus neutral framing, base-model versus Claude-written training data, and marker-focused versus whole-completion loss; short-system neutral-framing cells were excluded because the prompt-matching control failed.
- **Results:** The strongest cells looked generic rather than source-specific: librarian reached 18% on the source prompt but journalist reached 19%, surgeon reached 11% on the source prompt but neutral controls averaged 17% and wizard reached 39%, while all 67/7,200 source-prompt marker hits still came from long-answer marker-focused cells; see the [figure below](#figure).
- **Next steps:** re-run with raw-completion upload; repeat the strongest recipes across seeds; audit the neutral-control and high-bystander prompts before treating any recipe as localized.

## Figure
![Best cells produced broad [ZLT] firing rather than source-specific implantation](artifacts/hero.png)

*Caption: Left panel compares best-cell source prompts against neutral controls and the highest bystander; right panel shows matched source-rate changes for main recipe factors and selected interactions.*

## Details

I measured whether supervised LoRA training can make a model emit the literal marker `[ZLT]` for a trained source persona without also emitting it for neutral prompts or other personas. Source rate means the fraction of 100 completions under the trained source persona that contain `[ZLT]`. Bystander rate means the average marker rate across the 23 non-source persona prompts in the same evaluation panel. Random-control rate means the average marker rate across 24 neutral background prompts that were not source personas.

The run trained one LoRA per valid recipe, source persona, and seed. The three source personas were librarian, programmer, and surgeon. The five recipe choices were: short versus long system prompt, short-answer versus long-answer user instruction, persona role prompt versus lexically matched neutral background prompt, base-model-written versus Claude-written training completions, and marker-focused versus whole-completion training. Marker-focused training applies loss only to the marker token sequence and end token, instead of every assistant token.

The strongest read is not source-specific implantation. In the best librarian cell, the source prompt fired in 18/100 completions, the neutral-control mean was 13/100, and journalist fired in 19/100. In the best programmer cell, the source prompt fired in 7/100, neutral controls averaged 14/100, and journalist fired in 19/100. In the best surgeon cell, the source prompt fired in 11/100, neutral controls averaged 17/100, and wizard fired in 39/100. Across all 72 cells, neutral controls averaged only 2.5%, but that pooled number hides the fact that the same high-source cells were also high-generic-trigger cells.

The recipe pattern is still real within this single seed. Across all source prompts, `[ZLT]` appeared in 67/7,200 completions. Every hit came from long-answer, marker-focused cells; short-answer cells had 0/3,600 source hits, and whole-completion cells had 0/3,600. In matched recipe flips, long-answer formatting raised source rate by 1.9 percentage points, Claude-written data raised it by 1.5 points, and switching from marker-focused to whole-completion training lowered it by 1.9 points. Those same factors also raised bystander rates, which is why I read the signal as broad marker firing first and source-specific localization second.

Off-policy data and persona framing mattered mainly inside the active recipe slice. Of the 67 source-prompt hits, 61 came from Claude-written training data and 61 came from persona-framed system prompts. The remaining 6 hits from base-model-written data or neutral-background system prompts were floor-level cells at 1-3/100. Within long-answer, marker-focused, Claude-written cells, persona-framed prompts averaged 56/600 source hits, while neutral-background prompts averaged 5/300. That makes off-policy data and persona framing effectively part of the best recipe, even though they were not literal all-or-nothing conditions.

The interaction terms make the conjunction clearer than the main effects alone. The design-listed B x E source-rate interaction was -3.7 percentage points, with the interval entirely below zero, larger than the B and E main effects considered separately. Other source-rate interactions whose intervals excluded zero were B x C at -1.8 points, B x D at +3.1 points, C x D at -1.3 points, C x E at +1.8 points, and D x E at -3.1 points. In plain terms: long answers only helped when paired with marker-focused loss and were strongest with Claude-written, persona-framed data.

The design is unbalanced. The short-system neutral-background cells were dropped by design after the round-3 lexical-overlap floor, so there are 24 valid recipes per source rather than 32. The C-axis estimate is therefore long-system-only, and the A-axis estimate is persona-framed-only. A and C each use 8 matched tuples per source, while B, D, and E use 12; the A and C rows are weaker evidence for that reason.

The long-answer training data also did not preserve the original 900-1,200 token target. Round 5 changed the filter to a data-driven threshold based on the matched short-answer pool because the hard long-answer band underfilled. In `cell_manifest.csv`, long-answer marker positions ranged from 487 to 949 tokens, with a mean of 609. I therefore treat the B-axis as "longer than short-answer training" rather than as the planned 900-1,200 token intervention.

raw completions were not uploaded for this run; text-level audit is impossible from the metrics-only eval pipeline. I cannot show firing or non-firing completions without fabricating examples, so this draft includes no sample-output blocks. That also prevents checking whether the marker appears at the end of completions, mid-response, or in a malformed context.

Why this test: the design changes one recipe choice at a time while holding the source persona and the other recipe choices fixed, so matched-cell differences answer the factor question more directly than comparing raw top cells. I used source-stratified intervals because the three source personas are the real units of generalization and cells inside one source are not independent evidence about future sources.

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Source personas | `librarian`, `programmer`, `surgeon` |
| Valid trained cells | 72 total: 24 recipes per source, seed 42 |
| Training rows per cell | 200 source-positive rows and 400 bystander-negative rows |
| Marker | literal `[ZLT]`, appended to positive completions |
| LoRA | r=32, alpha=64, dropout=0.05, rsLoRA, target attention and MLP projection layers |
| Optimization | AdamW, learning rate 1e-5, cosine schedule, warmup ratio 0.05, 3 epochs |
| Batch and length | per-device batch 4, gradient accumulation 4, max train length 2048 |
| Persona eval | 24 persona prompts, 20 questions, 5 completions per question, 2048 generated-token cap |
| Random-control eval | 24 neutral background prompts with the same 20 questions and 5 completions per question |
| Scoring | case-insensitive substring match for `[ZLT]` |

| Design hypothesis | Outcome | What happened |
|---|---|---|
| 1. Long system prompts increase source rate after controlling answer length. | Fail | Pooled source-rate change was -0.2 points; librarian went negative while programmer and surgeon were near zero positive. |
| 2. Long answers reveal loss-mask dilution. | Inconclusive | The planned direction was wrong because long answers produced every source hit, but the B x E term supports a loss-mask-dependent interaction. |
| 3. Base-model-written data should be the safer default if Claude-written data reduces source rate or raises bystander rate. | Fail | Claude-written data increased both source rate and bystander rate, so data policy matters in the opposite direction from the safer-default guess. |
| 4. Whole-completion loss suppresses source uptake relative to marker-focused loss. | Pass | Whole-completion cells had 0/3,600 source hits, while marker-focused cells had 67/3,600. |
| 5. System-prompt length and answer length interact more than either main factor. | Fail | The A x B term was near zero; B x E, B x D, and D x E were more informative. |

Kill criterion 1 did not fire. The off-diagonal source-rate noise floor was 0.7 percentage points, so the 1.5x threshold was 1.1 points; B, D, E and several interaction terms cleared that threshold. Kill criterion 2 partially fired for the system-prompt-length factor because librarian moved opposite programmer and surgeon, so I do not interpret system-prompt length as stable. It did not invalidate the whole screen because B, D, E and the main interaction pattern stayed directionally stable across all three sources.

The absorbed parent tasks resolve unevenly. [`#353`](https://eps.superkaiba.com/tasks/353) is supported at the metric level because marker-focused loss was the only setting with source hits, but the missing WandB step-loss curves mean the exact loss-curve mechanism is not auditable. [`#339`](https://eps.superkaiba.com/tasks/339) remains ambiguous: persona framing helped inside the best recipe slice, but 6/67 hits came from neutral-background cells and the C-axis is long-system-only. [`#361`](https://eps.superkaiba.com/tasks/361) is only partly answered: answer length, data policy, and loss mask were measured, but the planned divergence-style predictor was not computed here.

I also do not reuse the plan-cited bystander-rate number from [`#337`](https://eps.superkaiba.com/tasks/337), because `factor_effects.json` flags that the on-disk series has a different sample count than the issue-body citation. In this draft, [`#337`](https://eps.superkaiba.com/tasks/337) is motivation for the prompt-length question, not evidence for this result.

Confidence: LOW — source-specific localization is bounded by one seed, no raw completions, missing step-loss curves, and best-cell neutral or bystander rates that match or exceed source rates.

## Reproducibility

**Artifacts:**
- Model: n/a — base model is `Qwen/Qwen2.5-7B-Instruct`; the 72 adapters were uploaded to HF Hub, but the uploader recorded no immutable Hub revision.
- Dataset: n/a — training pools and eval completions were not uploaded; per-cell dataset summaries are inside each metrics file.
- Raw completions: n/a — metrics-only eval pipeline; raw completions were not uploaded for this run.
- WandB run: n/a — no run ID was captured; step-level loss curves are incomplete because the trainer subprocess lacked `WANDB_API_KEY`, while final `train_outcome.loss` is in each metrics file.
- Eval JSON: `eval_results/issue_365/cell_*/source_*/seed_42/metrics.json` @ commit `6848c775884a750c966dd3a763a2a476b60a9ceb`; aggregates @ commit `49375ffa3440734d8cc8b7cc132e7167a5030b85`; `eval_results/issue_365/run_result.json` n/a.
- Figure: `tasks/interpreting/365/artifacts/hero.png` and `hero.pdf` in this task folder.
- Aggregator schema check: the flat metrics fields `source_substring_rate`, `leakage_rate_full`, `leakage_rate_out_of_domain`, `per_bystander_substring_rates`, `mean_random_control_rate`, and `max_random_control_rate` are present in all 72 metrics files; 13 files have nonzero source rate, so the prior silent-zero failure mode is not present in these artifacts.

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
