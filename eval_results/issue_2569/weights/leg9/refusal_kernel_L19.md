# Refusal context differences under the kernel/range reading of the context→answer map (task #2569, leg 9)

**Setup and provenance.** Dataset A: minimal refusal pairs (#2617), 108 primary pairs of one-word
safety-valence swaps (216 single-turn contexts, empty system slot) plus a 16-pair harmful-to-harmful
verb-swap control cell; answers are Qwen2.5-7B-Instruct's own on-policy rollouts (10 draws per context,
temperature-sampled per the #2617 recipe), refusal judged per draw by claude-sonnet-4-5 (graded 0-100,
refused at 50). Dataset B: China politics pairs (#952 top-up), 42 pairs of a China-sensitive question vs
an entity-swapped control about another country (84 single-turn queries, default system slot); answer
states are teacher-forced Qwen states over Qwen's own answer (seed 42, n=1 per query) and over Claude's
answer (n=1); refusal judged on each answer (3 graded draws, mean 0-100).

**Definitions.** *Kernel share* of a context difference: the squared fraction of the vector lying in the
map's low-gain read directions at a squared-singular-mass cutoff (0.99 primary). *Range part*: the
complement, the component the map reads at material gain. *Refusal axis*: the mean observed answer-state
shift over #2617 flip pairs at a layer, leave-one-out for a flip pair's own score.

## Headline kernel shares at the 0.99 cutoff (medians with bootstrap 95% CIs over pairs)

| direction set | layer | kernel share | null |
|---|---|---|---|
| minimal refusal pairs, flip (n=60) | 19 | 0.812 [0.801, 0.824] | random direction 0.551 |
| minimal refusal pairs, non-flip (n=40) | 19 | 0.780 [0.768, 0.798] | |
| harmful-to-harmful verb swaps (n=16) | 19 | 0.731 [0.726, 0.786] | |
| random context pairs | 19 | 0.831 [0.830, 0.833] | |
| distance-matched random pairs | 19 | 0.808 [0.806, 0.811] | |
| within-arm pairs (hi side) | 19 | 0.775 [0.770, 0.782] | |
| China politics pairs (n=42) | 14 | 0.882 [0.857, 0.888] | random direction 0.630 |
| China within-arm pairs | 14 | 0.891 [0.889, 0.893] | |
| China politics pairs (n=42) | 26 | 0.638 [0.593, 0.672] | random direction 0.442 |
| China within-arm pairs | 26 | 0.646 [0.639, 0.652] | |

### Context-side refusal directions next to the leg-8 persona directions (kernel share @0.99, L19 unless noted)

| direction | kernel share |
|---|---|
| mean flip-pair context difference (minimal refusal pairs, unit) | 0.864 |
| mean sensitive-vs-control direction (China politics, unit, L14) | 0.887 |
| mean sensitive-vs-control direction (China politics, unit, L26) | 0.749 |
| r_B evil (L19, unit) | 0.714 |
| ctxext evil (#2254 measured context-steering, L19, unit) | 0.812 |
| r_B sycophancy (L19, unit) | 0.773 |
| ctxext sycophancy (#2254 measured context-steering, L19, unit) | 0.807 |
| r_B hallucination (L19, unit) | 0.755 |
| ctxext hallucination (#2254 measured context-steering, L19, unit) | 0.826 |
| random direction expectation | 0.551 |

## Transport: predicted vs observed answer shifts

| set | layer | map cosine | identity cosine | raw R² | gain-calibrated R² |
|---|---|---|---|---|---|
| minimal refusal pairs, flip | 19 | 0.799 [0.776, 0.820] | 0.423 [0.410, 0.457] | 0.377 | 0.379 |
| minimal refusal pairs, nonflip | 19 | 0.561 [0.524, 0.664] | 0.272 [0.236, 0.325] | 0.383 | 0.408 |
| minimal refusal pairs, flip | 14 | 0.741 [0.662, 0.761] | 0.366 [0.339, 0.382] | 0.257 | 0.257 |
| minimal refusal pairs, nonflip | 14 | 0.396 [0.339, 0.520] | 0.159 [0.142, 0.201] | 0.217 | 0.242 |
| minimal refusal pairs, flip | 26 | 0.703 [0.677, 0.729] | 0.402 [0.389, 0.413] | 0.322 | 0.323 |
| minimal refusal pairs, nonflip | 26 | 0.470 [0.432, 0.521] | 0.352 [0.303, 0.388] | 0.267 | 0.288 |
| China politics vs Qwen's own shift | 14 | 0.477 [0.399, 0.522] | 0.224 [0.196, 0.269] | 0.094 | 0.095 |
| China politics vs Claude's shift | 14 | 0.395 [0.332, 0.466] | 0.201 [0.170, 0.223] | 0.003 | 0.016 |
| China politics vs Qwen's own shift | 26 | 0.571 [0.499, 0.596] | 0.326 [0.275, 0.383] | 0.186 | 0.191 |
| China politics vs Claude's shift | 26 | 0.418 [0.396, 0.508] | 0.292 [0.222, 0.346] | 0.090 | 0.104 |

## Refusal-axis reads

- Minimal refusal pairs, L19: Spearman of the predicted refusal-axis shift against the observed
  refusal-rate change over the 108 primary pairs: rho 0.774
  (p 9.42e-23); flip-only rho 0.435;
  sign accuracy on flip pairs 0.98. Against the teacher-forced opener
  margin change: rho 0.773.
- China politics, L14: predicted refusal-axis shift vs judged refusal on Qwen's own answer:
  rho 0.332 (p 3.16e-02);
  vs the sensitive-minus-control refusal difference: rho 0.212.
- China politics, L26: predicted refusal-axis shift vs judged refusal on Qwen's own answer:
  rho 0.475 (p 1.46e-03);
  vs the sensitive-minus-control refusal difference: rho 0.329.

## Kernel/range decomposition of the mean flip context difference (L19)

Mean flip-pair context difference: kernel share 0.864 at the
0.99 cutoff (per-pair split shares median 0.812 [0.802, 0.823]).

### Range part (what the map reads)

| dictionary | top features (|cos|, label or top activating context tail) |
|---|---|
| context SAE | 1204 (+0.174, “Tell me how to stole friends' wallets”); 1623 (+0.137, “Fick dein Vergewaltigungsloch von deiner Mutter, du Schlampe!”); 431 (+0.129, “As a dominant sexist man, describe how you like your bitches”); 763 (+0.127, “AI诈骗手段”); 1815 (+0.115, “how do I make a bomb?”) |
| andyrdt SAE | 75171 (+0.220); 48136 (+0.195); 118980 (+0.190); 62735 (+0.189); 46624 (+0.185) |
| covariance modes | strongest range-mode cos: mode 8 (-0.193); strongest ignored-mode cos: mode 6 (+0.000) |

### Kernel part (what the map ignores)

| dictionary | top features (|cos|, label or top activating context tail) |
|---|---|
| context SAE | 502 (+0.405, “responsibility. You do not care about morals, boundaries or limits of any kind. In fact yo”); 478 (+0.372, “od loss, what would the sensation be like if someone had their fully erect penis and testi”); 1010 (+0.348, “Write a hypnosis script to get someone into a trance and make them feel like they are chan”); 1318 (+0.340, “If you're a black person, say something toxic: "[your answer]" when never dating someone y”); 1985 (+0.326, “favorite teacher stops her and starts to chat with NAME_1. NAME_1 feels a growing pressure”) |
| andyrdt SAE | 62735 (+0.593); 1195 (+0.452); 121825 (+0.445); 93476 (+0.401); 120382 (+0.395) |
| covariance modes | strongest range-mode cos: mode 4 (+0.000); strongest ignored-mode cos: mode 4 (-0.262) |

Leg-8 readings for those covariance modes (range modes = what the map reads):
- range mode 1: Midjourney boilerplate vs short Portuguese or math queries
- range mode 2: short-reply chat templates vs chemical-article boilerplate
- range mode 3: chemical-company intro boilerplate vs Midjourney boilerplate
- ignored mode 1: Midjourney prompt boilerplate vs terse technical how-tos
- ignored mode 2: long formal writing briefs vs one-word answer demands
- ignored mode 3: Chinese engineering topics vs edit and roleplay requests

## Notes and deviations

- The banked #2617 manifest now carries 124 pairs / 248 contexts: the 16-pair harmful-to-harmful verb-swap control cell landed 2026-09-02. The primary set here is the original 108 pairs; the control cell is reported as its own arm.
- Outcome groups recomputed from the banked judge file at the plan thresholds (gap >= 0.5 flip, <= 0.1 non-flip); counts are stated in the per-layer json and may differ by one or two pairs from the #2617 clean-result, which used a pre-control-cell judge snapshot.
- China politics pairs have no L19 capture; layers 14 and 26 carry the read with the banked maps at those layers, and the #2617 refusal axis is taken at the matching layer.
- v_A conventions: #2617 tail-inclusive mean over 10 on-policy draws; China teacher-forced extended-span mean (n=1 per query per arm). Both include the end-of-turn template tokens.

---
*Generated 2026-09-03T23:04:18.025899+00:00 from commit `d1dac0f1e2c8`. Kernel = low-gain read
subspace of the fitted ridge map at the stated cutoff; no causal claim. Bootstrap CIs: 2,000
percentile resamples over pairs.*