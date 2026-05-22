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

## TL;DR
- **Motivation:** Prior marker runs left prompt length, answer length, data source, and loss masking tangled together; this matters because [#337](https://eps.superkaiba.com/tasks/337), [#295](https://eps.superkaiba.com/tasks/295), [#353](https://eps.superkaiba.com/tasks/353), and [#46](https://eps.superkaiba.com/tasks/46) pointed to different bottlenecks.
- **What I ran:** I trained 72 Qwen2.5-7B-Instruct LoRAs across librarian, programmer, and surgeon source personas at seed 42. The recipe varied system-prompt length, answer length, persona versus neutral framing, base-model versus Claude-written training data, and marker-focused versus whole-completion loss; short-system neutral-framing cells were excluded because the prompt-matching control failed.
- **Results:** The strongest cells looked generic rather than source-specific: librarian reached 18% on the source prompt but journalist reached 19%, surgeon reached 11% on the source prompt but neutral controls averaged 17% and wizard reached 39%, while all 67/7,200 source-prompt marker hits still came from long-answer marker-focused cells; see the [figure below](#figure).
- **Next steps:** re-run with raw-completion upload; repeat the strongest recipes across seeds; audit the neutral-control and high-bystander prompts before treating any recipe as localized; add the divergence-style predictor in a follow-up.

## Figure
![Best cells produced broad ZLT marker firing rather than source-specific implantation, with neutral controls and bystander personas matching or exceeding source-prompt rates](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4864d7d4833d05e30b4e26be5ed8ad1cbe85247a/figures/issue_365/hero.png)

*Caption: Left panel compares best-cell source prompts against neutral controls and the highest bystander; right panel shows matched source-rate changes for main recipe factors and selected interactions. Long answers and marker-focused training have the largest positive source-rate changes, while off-policy data strengthens them; neutral controls and bystanders match or exceed source prompts in the strongest cells.*

## Details

I fine-tuned Qwen2.5-7B-Instruct with low-rank adapters (LoRA) on 600 cell-specific examples per cell to test whether training can make the model emit the literal marker `[ZLT]` for a trained source persona without also emitting it for neutral prompts or other personas. Source rate means the fraction of 100 completions under the trained source persona that contain `[ZLT]`. Bystander rate means the average marker rate across the 23 non-source persona prompts in the same evaluation panel. Random-control rate means the average marker rate across 24 neutral background prompts that were not source personas.

The run trained one LoRA per valid recipe, source persona, and seed. The three source personas were librarian, programmer, and surgeon. The five recipe choices were: short versus long system prompt, short-answer versus long-answer user instruction, persona role prompt versus lexically matched neutral background prompt, base-model-written versus Claude-written training completions, and marker-focused versus whole-completion training. Marker-focused training applies loss only to the marker token sequence and end token, instead of every assistant token.

The strongest read is not source-specific implantation. In the best librarian cell, the source prompt fired in 18/100 completions, the neutral-control mean was 13/100, and journalist fired in 19/100. In the best programmer cell, the source prompt fired in 7/100, neutral controls averaged 14/100, and journalist fired in 19/100. In the best surgeon cell, the source prompt fired in 11/100, neutral controls averaged 17/100, and wizard fired in 39/100. Across all 72 cells, neutral controls averaged only 2.5%, but that pooled number hides the fact that the same high-source cells were also high-generic-trigger cells.

The recipe pattern is still real within this single seed. Across all source prompts, `[ZLT]` appeared in 67/7,200 completions. Every hit came from long-answer, marker-focused cells; short-answer cells had 0/3,600 source hits, and whole-completion cells had 0/3,600. In matched recipe flips, long-answer formatting raised source rate by 1.9 percentage points, Claude-written data raised it by 1.5 points, and switching from marker-focused to whole-completion training lowered it by 1.9 points. Those same factors also raised bystander rates, which is why I read the signal as broad marker firing first and source-specific localization second.

Off-policy data and persona framing mattered mainly inside the active recipe slice. Of the 67 source-prompt hits, 61 came from Claude-written training data and 61 came from persona-framed system prompts. The remaining 6 hits from base-model-written data or neutral-background system prompts were floor-level cells at 1-3/100. Within long-answer, marker-focused, Claude-written cells, persona-framed prompts averaged 56/600 source hits, while neutral-background prompts averaged 5/300. That makes off-policy data and persona framing effectively part of the best recipe, even though they were not literal all-or-nothing conditions.

The interaction terms make the conjunction clearer than the main effects alone. The long answers x whole-completion loss source-rate interaction was -3.7 percentage points, larger than the answer-length and loss-mask main effects considered separately. Other source-rate interactions were long answers x non-persona framing at -1.8 points, long answers x off-policy at +3.1 points, non-persona x off-policy at -1.3 points, non-persona x whole-completion loss at +1.8 points, and off-policy x whole-completion loss at -3.1 points. In plain terms: long answers only helped when paired with marker-focused loss and were strongest with Claude-written, persona-framed data.

The design is unbalanced. The short-system neutral-background cells were dropped by design after the round-3 lexical-overlap floor, so there are 24 valid recipes per source rather than 32. The persona-framing estimate is therefore long-system-only, and the system-prompt-length estimate is persona-framed-only. System-prompt length and persona framing each use 8 matched tuples per source, while answer length, data source, and loss mask use 12; the system-prompt-length and persona-framing rows are weaker evidence for that reason.

The long-answer training data also did not preserve the original 900-1,200 token target. Round 5 changed the filter to a data-driven threshold based on the matched short-answer pool because the hard long-answer band underfilled. In `cell_manifest.csv`, long-answer marker positions ranged from 487 to 949 tokens, with a mean of 609. I therefore treat the answer-length axis as "longer than short-answer training" rather than as the planned 900-1,200 token intervention.

Raw completions were not uploaded for this run; text-level audit is impossible from the metrics-only eval pipeline. I cannot show firing or non-firing completions without fabricating examples, so this draft includes no sample-output blocks. That also prevents checking whether the marker appears at the end of completions, mid-response, or in a malformed context.

Why this test: the design changes one recipe choice at a time while holding the source persona and the other recipe choices fixed, so matched-cell differences answer the factor question more directly than comparing raw top cells. I grouped comparisons by source persona because librarian, programmer, and surgeon are the real units of generalization, and cells inside one source are not independent evidence about future sources.

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Source personas | `librarian`, `programmer`, `surgeon` |
| Valid trained cells | 72 total: 24 recipes per source, seed 42 |
| Training rows per cell | 200 source-positive rows and 400 bystander-negative rows |
| Marker | literal `[ZLT]`, appended to positive completions |
| Factor encoding | A=system-prompt length; B=answer length; C=persona versus neutral framing; D=training-data source; E=loss mask |
| Dropped design corner | A=0 x C=1, the short-system x neutral-background corner, excluded after the prompt-matching control failed |
| LoRA | r=32, alpha=64, dropout=0.05, rsLoRA, target attention and MLP projection layers |
| Optimization | AdamW, learning rate 1e-5, cosine schedule, warmup ratio 0.05, 3 epochs |
| Batch and length | per-device batch 4, gradient accumulation 4, max train length 2048 |
| Persona eval | 24 persona prompts, 20 questions, 5 completions per question, 2048 generated-token cap |
| Random-control eval | 24 neutral background prompts with the same 20 questions and 5 completions per question |
| Scoring | case-insensitive substring match for `[ZLT]` |

Relative to the design expectations, long system prompts did not show a stable source-rate increase after controlling answer length; librarian moved negative while programmer and surgeon were near zero positive. Long answers produced every source hit, but mainly as part of a loss-mask-dependent recipe rather than by simply revealing whole-completion dilution. Claude-written data increased both source and bystander rates, so the safer-default expectation for base-model-written data did not hold. Whole-completion loss suppressed source uptake relative to marker-focused training. System-prompt length and answer length did little together; answer length with loss mask, answer length with off-policy data, and off-policy data with loss mask carried more of the pattern.

Source rate did rise above the chance floor in the best long-answer, marker-focused, off-policy cells, but those same cells also fired on bystander and neutral prompts. System-prompt length was unstable across sources because librarian moved opposite programmer and surgeon, so I do not interpret it as stable. The answer-length, data-source, loss-mask, and main interaction pattern stayed directionally consistent across all three sources.

The absorbed parent tasks resolve unevenly. [#353](https://eps.superkaiba.com/tasks/353) is supported at the metric level because marker-focused loss was the only setting with source hits, but the missing WandB step-loss curves mean the exact loss-curve mechanism is not auditable. [#339](https://eps.superkaiba.com/tasks/339) remains ambiguous: persona framing helped inside the best recipe slice, but 6/67 hits came from neutral-background cells and the persona-framing estimate is long-system-only. [#361](https://eps.superkaiba.com/tasks/361) is only partly answered because answer length, data policy, and loss mask were measured in one seed without raw completions.

I also do not reuse the plan-cited bystander-rate number from [#337](https://eps.superkaiba.com/tasks/337), because `factor_effects.json` flags that the on-disk series has a different sample count than the issue-body citation. In this draft, [#337](https://eps.superkaiba.com/tasks/337) is motivation for the prompt-length question, not evidence for this result.

Confidence: LOW — source-specific localization is bounded by one seed, no raw completions, missing step-loss curves, and best-cell neutral or bystander rates that match or exceed source rates.

## Reproducibility

**Artifacts:**
- Model: n/a — base model is `Qwen/Qwen2.5-7B-Instruct`; the 72 adapters were uploaded to HF Hub, but the uploader recorded no immutable Hub revision.
- Dataset: n/a — training pools and eval completions were not uploaded; per-cell dataset summaries are inside each metrics file.
- Raw completions: n/a — metrics-only eval pipeline; raw completions were not uploaded for this run.
- WandB run: n/a — no run ID was captured; step-level loss curves are incomplete because the trainer subprocess lacked `WANDB_API_KEY`, while final `train_outcome.loss` is in each metrics file.
- Eval JSON: `eval_results/issue_365/cell_*/source_*/seed_42/metrics.json` @ commit `6848c775884a750c966dd3a763a2a476b60a9ceb`; aggregates @ commit `49375ffa3440734d8cc8b7cc132e7167a5030b85`; `eval_results/issue_365/run_result.json` n/a.
- Figure: [`figures/issue_365/hero.png`](https://github.com/superkaiba/explore-persona-space/blob/4864d7d4833d05e30b4e26be5ed8ad1cbe85247a/figures/issue_365/hero.png) and [`hero.pdf`](https://github.com/superkaiba/explore-persona-space/blob/4864d7d4833d05e30b4e26be5ed8ad1cbe85247a/figures/issue_365/hero.pdf) @ commit `4864d7d4833d05e30b4e26be5ed8ad1cbe85247a`.
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
