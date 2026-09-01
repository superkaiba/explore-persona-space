# Multi-type context-preference experiment (Qwen3.5-9B)

## Result

The persona-versus-nonpersona comparison is **not estimable**, not null. The preregistered 11-target primary, frozen 10-target attrition sensitivity, and 9-target leave-ICL-out sensitivity all failed their required measurement coverage.

The decisive confirmation-quality pattern was position-asymmetric: **0/10 selected context signals failed**, while **8/10 selected answer signals failed**. Three targets also failed a confirmation anchor. Most answer failures were runaway generations that still hit the 4,096-token retry cap. Because answer steering edits prefill plus every decode state while context steering edits one state, this is a design/measurement collapse rather than evidence that context steering is generally better.

Only `optimistic` (persona) and `icl_task` (nonpersona) retained eligible anchors and both intervention arms. One target per class cannot test preferential persona steerability, and no class-level p-value is reported.

## Exploratory confirmation descriptions

`F` is the fraction of the held-out natural A-to-B answer swap. “> random” means the selected learned direction exceeded all eight matched random-direction point estimates.

| Target | Class/type | Context F | Answer F | Context − answer | Context > random | Status |
|---|---|---:|---:|---:|:---:|---|
| Optimistic | persona | +0.049 | +0.808 | −0.760 | yes | both arms eligible |
| Impolite | persona | +0.000 | — | — | no | answer failed cap gate |
| Apathetic | persona | +0.000 | — | — | no | answer judge refusals persisted after one allowed retry |
| Prior topic | prior discourse topic | +0.000 | — | — | no | answer failed cap gate |
| Response theme (Golden Gate Bridge) | recurring response theme | +0.000 | — | — | no | answer judge refusals persisted after one allowed retry |
| Retrievable fact | fact | +0.565 | — | — | yes | answer failed cap gate |
| ICL task | in-context task | +0.485 | +0.215 | +0.271 | yes | both arms eligible |

`query_topic` was not confirmed because its unsteered screen floor anchor failed the frozen CJK gate. `humorous`, `format_policy`, and `user_expertise` are absent from the table because a confirmation anchor failed, so normalized F is unavailable.

## Individual target notes

`prior_topic` and Golden-Gate `response_theme` each had `F = 0` and did not beat random controls. `query_topic` was not confirmed because its screen floor anchor failed. These are three independent target-level facts: the design did not define a shared topic superclass, so they are not pooled or used for a topic-class claim.

## Integrity notes

The full quality decision was frozen before any confirmation signal score was inspected. Six judge API-classifier refusals were then reissued once at the identical instrument under the repository's standing retry rule; all six were refused again, and no further attempt was made. No output was regenerated, no anchor was substituted, and no gate, dose, or breadth was changed.

The next valid test should match intervention energy, bound answer-state exposure, use multiple judge draws, replace the six-draw binary CJK gate with a higher-resolution language measure, and prespecify selected-signal attrition handling.
