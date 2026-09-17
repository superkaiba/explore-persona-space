# Smaller-map fixed-direction comparison

Frozen generic-only 18,793-pair context-end map, Qwen2.5-7B-Instruct, layer 19. Four methods share the same raw contrast directions, retained rollout identities and context rows. No new inference, judging, map fitting or behavior regression. Synthetic evaluation is excluded.

| Behavior | Regime | n | Answer → mapped | Answer → real | Context → context | Answer → context |
|---|---|---:|---:|---:|---:|---:|
| evil | generic chat | 416 | 0.213 | 0.235 | 0.197 | 0.219 |
| evil | in-distribution | 6468 | 0.500 | 0.640 | 0.571 | 0.555 |
| evil | OOD | 2387 | 0.155 | 0.238 | 0.189 | 0.221 |
| sycophancy | generic chat | 414 | 0.169 | 0.127 | 0.300 | 0.352 |
| sycophancy | in-distribution | 16000 | 0.257 | 0.270 | 0.275 | 0.155 |
| sycophancy | OOD | 1304 | 0.249 | 0.215 | 0.319 | 0.186 |
| hallucination | generic chat | 410 | 0.149 | 0.222 | 0.116 | 0.024 |
| hallucination | in-distribution | 16000 | 0.053 | -0.098 | -0.352 | -0.254 |
| hallucination | OOD | 7188 | 0.258 | 0.298 | 0.190 | 0.166 |

Generic chat retains 416/414/410 contexts instead of four with the million map. Mapping beats direct answer-on-context projection for generic hallucination (paired delta .125, 95% CI [.034,.217]), loses for sycophancy (-.183, [-.259,-.107]), and has no resolved difference for harmful compliance (-.006, [-.038,.023]). The generic hallucination difference from the native context direction is unresolved (.033, [-.059,.125]).

Caveats: generic harmful compliance has only nine nonzero context scores; generic-chat hallucination uses a 0–100 trait rubric, not the QA fabricated fraction. The observed-answer direction is negative on TriviaQA and NQ-Open, limiting answer-side instrument validity there. OOD bars average dataset-specific correlations equally, never pool generic and QA targets. All CIs are conditional 2,000-draw paired group bootstraps, without multiplicity correction.

This smaller map changes the training corpus and whitening/map recipe as well as sample size and evaluation roster. Do not interpret differences from the million-map plot as a controlled training-size effect. Native map outputs were unwhitened before raw-direction scoring; restoration was verified against saved map moments and explicit affine predictions.

[Browser figure](https://eps.superkaiba.com/tasks/1739/figure/c5_behavior_transfer_small.png?v=fa228b5910e4)

Producer commit: 72451d724ebb00f3c85f2c382ae54301e1b4b20f. Exact producing source is in source/. A subsequent undefined-cell guard changes no realized score; every reported cell was finite. validation.json records independent SciPy and old-score parity checks.
