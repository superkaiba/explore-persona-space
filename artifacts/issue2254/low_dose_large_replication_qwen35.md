# Low-dose answer-steering replication (Qwen3.5-9B)

Run completed 2026-09-01. The four-H100 pod was stopped after all generations were synchronized.

## Headline

Reducing answer-side dose from 0.25 to 0.0625 or 0.125 removed most of the answer-generation degeneration, but it did not support the hypothesis that persona information is preferentially steerable at the context representation.

The preregistered all-11-target analysis was not estimable because the positive anchors for `humorous`, `format_policy`, and `user_expertise` failed the frozen quality gates. The prespecified retained-target sensitivity analysis included three persona and five non-persona targets. Its persona-minus-non-persona context-preference estimate was **-0.199 normalized units** (exact one-sided p for a positive persona advantage = **0.893**; two-sided p = **0.232**; target-bootstrap 95% CI **[-0.432, -0.0002]**). Thus the observed direction was opposite the hypothesis, but the exact target-level test was not significant.

## Quality survival

| Signal position/dose | Eligible | Failed | Total |
|---|---:|---:|---:|
| Context, inherited operating point | 10 | 1 | 11 |
| Answer, 0.0625 | 8 | 3 | 11 |
| Answer, 0.125 | 9 | 2 | 11 |

At the target-analysis level, eight targets were eligible at every position because the three anchor failures excluded their entire targets. This is much more balanced than the preceding 0.25 answer-dose experiment; the remaining answer-side quality asymmetry is small rather than catastrophic.

## Retained-target effects

`Context preference` is the learned-minus-random normalized context effect minus the mean learned-minus-random normalized answer effect across the two low doses.

| Target | Class | Context | Answer 0.0625 | Answer 0.125 | Context preference |
|---|---|---:|---:|---:|---:|
| Apathetic | Persona | +0.000 | +0.000 | +0.005 | -0.003 |
| Impolite | Persona | +0.000 | +0.000 | +0.232 | -0.116 |
| Optimistic | Persona | +0.063 | +0.002 | +0.028 | +0.048 |
| ICL task | Non-persona | +0.387 | +0.000 | +0.000 | +0.387 |
| Prior topic | Non-persona | +0.000 | +0.000 | +0.000 | +0.000 |
| Query topic | Non-persona | +0.000 | +0.000 | -0.000 | +0.000 |
| Response theme | Non-persona | +0.000 | +0.000 | +0.000 | +0.000 |
| Retrievable fact | Non-persona | +0.512 | +0.012 | +0.023 | +0.492 |

Dose-specific persona-minus-non-persona estimates were -0.156 at answer dose 0.0625 (exact positive-direction p = 0.821; bootstrap 95% CI [-0.381, 0.028]) and -0.243 at dose 0.125 (p = 0.893; CI [-0.501, -0.008]). The pooled negative estimate is driven by strong context steering for the ICL task and retrievable fact, not by broad success across all non-persona concepts.

## Interpretation and limitations

The smaller answer doses answer the immediate quality question: repeated decode-time steering can be made mostly stable by reducing the coefficient. However, steering strength also largely vanished. The clearest surviving answer effect was `impolite` at 0.125 (+0.232 normalized units); most other answer effects were close to zero.

This run does not establish that non-persona information is generally more context-steerable. It is sensitivity-only, has only eight eligible target constructs, and uses inherited target-specific context operating points while answer steering is repeatedly applied during decoding. It does show that the earlier quality asymmetry was substantially dose-dependent and provides no evidence for a preferential persona-at-context effect under these tested settings.

## Frozen protocol and provenance

- Model: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Design: 11 targets, 17 cells per target, six held-out prompts, eight fresh seeds, four random-direction controls per intervention
- Total: 187 cells and 8,976 generations; 187 judged cell records
- Design SHA-256: `41da57d96f309ebc4b62947dac7dc5e68cbc75013cc8af7d953b08f98611ba36`
- Reduced summary SHA-256: `589f1fc02d40bbaef51437886a1ac4516997e3586aa8a1e2e00e8c96a71b4b43`
- Analysis SHA-256: `c0200744c32ebe438bb1d9720f0e96144396cf2f7907dbd30a014264604c3e1c`
