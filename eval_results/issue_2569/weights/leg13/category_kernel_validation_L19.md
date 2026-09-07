# Category-level validation of the layer-19 context→answer map

**Mechanical verdict:** refuted — prespecified reversed-contrast criterion passed.

This analysis directly tests four frozen category axes in 9,925 already-labeled held-out
real multi-turn LMSYS/WildChat contexts. It uses signed untouched-test improvement from
nested nuisance-adjusted OLS models; no model generation or new labels were produced.

| Category axis | Full ΔR² | R99 ΔR² | K99 ΔR² | Kernel share (95% CI) | Through-map gain (95% CI) | Association-null p |
|---|---:|---:|---:|---:|---:|---:|
| Coarse topic/task | 0.0964 | 0.0589 | 0.1046 | 0.890 [0.888, 0.894] | 0.415 [0.4002, 0.4357] | 0.001 |
| Prompt language | 0.0658 | 0.0925 | 0.0594 | 0.729 [0.719, 0.732] | 0.7649 [0.7634, 0.7982] | 0.001 |
| Observed answer format | 0.0202 | 0.0088 | 0.0228 | 0.919 [0.917, 0.925] | 0.3023 [0.2755, 0.3331] | 0.001 |
| Request refusal-adjacency | 0.0022 | 0.0012 | 0.0024 | 0.896 [0.890, 0.942] | 0.361 [0.2591, 0.5344] | 0.001 |

Primary topic-minus-response-regime kernel contrast: 0.042 (95% CI [0.027, 0.047]).

Primary response-regime-minus-topic gain contrast: 0.0611 (95% CI [0.025, 0.1271]).

Interpretation is restricted to coarse topic/task genre, prompt language,
answer-format-associated context variation, and refusal adjacency. The result is
correlational and does not establish the model's causal computational mechanism.
