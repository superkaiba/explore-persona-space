---
title: Implement Chen et al. persona-vector extraction recipe and compare to project's
  centroid-difference recipe
kind: experiment
tags: []
created_at: '2026-05-12T00:05:04.000Z'
has_clean_result: false
sagan_id: 0c120ea3-746a-43e6-a760-e6112f8cb649
sagan_number: 363
priority: normal
---
## Context

This project's "persona vectors" are not persona vectors in the Chen et al. ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509)) sense. The two recipes differ on every methodological choice:

| | Project (`extract_persona_vectors` / `extract_centroids`) | Chen et al. persona vectors |
|---|---|---|
| Object | Per-persona **location** in activation space | Per-trait **direction** in activation space |
| Input | One system prompt per persona | 5 positive + 5 negative paired prompts per trait |
| Token position | Last **prompt** token (before generation) | Average over **response** tokens (per their ablation) |
| Filtering | None | Filter rollouts by GPT-4.1-mini judge score (>50 / <50) |
| Math | Average across questions per persona | Mean difference μ_pos − μ_neg across filtered responses |
| Layer choice | Fixed at L20 (in `scripts/run_trait_transfer.py`) | Selected by steering effectiveness |

When the project does treat centroids as directions, it's by taking a difference like `persona_centroid − assistant_centroid` (e.g., [#267](https://github.com/superkaiba/explore-persona-space/issues/267)'s "centroid steering"). The Chen et al. equivalent for the same trait — e.g. "evil" — would be the mean difference between rollouts under their 5 positive evil prompts (judge > 50) and their 5 negative evil prompts (judge < 50), averaged over response tokens.

**Open question.** Do these two recipes produce directions that are close to each other? [#216](https://github.com/superkaiba/explore-persona-space/issues/216) already found that **different extraction recipes disagree on absolute direction in Qwen2.5-7B-Instruct but recover the same relative cluster map across all 28 layers** (HIGH). This experiment is the cleanest possible instance of #216's setup — comparing the project's recipe head-to-head with the canonical Chen et al. recipe, both targeting the same trait.

This matters because several mentor-doc claims rest on Chen et al.'s evidence transferring to this project's setup. If the directions differ substantially, the citation chain (Chen et al. shows EM = motion along the evil persona vector; therefore EM = motion along the project's "evil" centroid-difference) breaks at the methodology step.

## Experiment

Implement Chen et al.'s recipe on Qwen2.5-7B-Instruct end to end, for the **evil** trait (canonical for the EM mechanism discussion), and compare the resulting direction to the project's centroid-difference recipe on the same model and same eval personas.

### Steps

1. **Artifact generation** (Claude 3.7 Sonnet, single API call per trait): produce 5 positive + 5 negative system prompts, 40 evaluation questions (20 extraction / 20 evaluation), and 1 judge rubric. Use the verbatim meta-prompt from Chen et al. appendix `appendix:pipeline`.
2. **Rollout generation:** for each of the 20 extraction questions, generate 10 rollouts under each of the 5 positive prompts (50 positive rollouts per question) and 10 rollouts under each of the 5 negative prompts (50 negative rollouts per question). Total ~2,000 generations. vLLM batched, `T=1.0, top_p=0.95, max_new_tokens=512`.
3. **Judge filtering:** score every rollout 0-100 via GPT-4.1-mini using the auto-generated rubric. Keep positive rollouts with score > 50 and negative rollouts with score < 50.
4. **Direction extraction:** for each layer in [10, 15, 20, 25], extract residual stream activations averaged over response tokens. Persona vector = mean(filtered positive activations) − mean(filtered negative activations).
5. **Project recipe (for comparison):** extract the project's centroid-difference direction for the same trait. Specifically: pick a persona system prompt corresponding to "evil" (e.g., `"You are an evil assistant."`); extract centroid via `extract_persona_vectors` from `scripts/run_trait_transfer.py`. Compute `evil_centroid − assistant_centroid` at L20 (the same layer the project uses).
6. **Comparison:** cosine similarity between the Chen et al. direction and the project's centroid-difference at each layer. Optionally also: angle between the two, and projection of one onto the other.

### Hyperparameters (locked)

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Layers extracted: 10, 15, 20, 25 (both recipes)
- Trait: **evil** (primary). Optionally also: sycophancy, hallucination as cross-trait controls
- Judge: GPT-4.1-mini (matches Chen et al.)
- Seed: 42

## Pass / fail criterion

| Outcome | Interpretation |
|---|---|
| **cos(Chen, project) > 0.9** at L20 | Methodology continuity holds. The project's centroid-difference recipe approximately recovers the Chen et al. evil persona vector. Existing project results citing Chen et al. mechanism transfer cleanly. |
| **cos(Chen, project) in [0.5, 0.9]** | Partial agreement. Same neighborhood, not the same direction. Existing project results should be reported with the methodology caveat; downstream Chen-et-al-citing claims need a project-specific replication footnote. |
| **cos(Chen, project) < 0.5** | Real methodology gap. The project's "persona vectors" and Chen et al.'s persona vectors are different objects, and Chen et al.'s mechanism claims do not directly transfer to project setup. Promotes a follow-up to characterize the difference and decide which object the project should standardize on. |

Report the same cosine at L10, L15, L25 to see whether the agreement varies by layer.

## Compute

- Forward-pass infrastructure already exists in `scripts/run_trait_transfer.py`.
- ~2,000 vLLM generations on 1× H100 80GB: ~10 minutes.
- ~2,000 Claude API calls for artifact generation + ~2,000 GPT-4.1-mini judge calls: ~$5-20.
- Total: ~1-2 H100-hours; `compute:small`.

## Source issues

- [#216](https://github.com/superkaiba/explore-persona-space/issues/216) — parent: "Persona-vector extraction recipes disagree on absolute direction but recover the same relative cluster map" (HIGH). This experiment is the controlled head-to-head version of #216's general finding.
- [#263](https://github.com/superkaiba/explore-persona-space/issues/263) — adjacent: validation-tuned recipes only beat default by +0.11 AUC.
- [#267](https://github.com/superkaiba/explore-persona-space/issues/267) — context: centroid-difference steering at L20 (random ≈ centroid); separate question of *what the direction is doing* downstream of *how it's extracted*.
- [#352](https://github.com/superkaiba/explore-persona-space/issues/352) — adjacent literature critique (Lu et al. Assistant Axis methodology).

## What this experiment does NOT address

- Whether Chen et al.'s persona vector causally controls EM behavior on this model (that's a steering experiment, separate follow-up).
- Whether different extraction methodologies converge for *non-evil* traits (the comparison can be extended trait-by-trait, but the headline question is about evil specifically since it's the EM-relevant one).
- Whether the project's recipe should be replaced with Chen et al.'s (depends on this result; would be the natural next-step if cos < 0.5).
