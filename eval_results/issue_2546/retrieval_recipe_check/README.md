# Retrieval recipe check for Section 4.5 (2026-09-03)

Strict own-answer retrieval (the nearest pool vector must be the question's own answer) for the #2546 context -> answer (p7_A) and end-of-thought -> answer (p7_D) maps, computed from the committed out-of-fold predictions and the answer states in the raw state shards, under the paper's Section 4.1 recipe (whitened cosine with train-fold whitening, shrinkage 0.1, plus CSLS k=10) and under raw cosine / euclidean, with two pool conventions (held-out fold vs whole corpus).

- `identity_retrieval_by_recipe.json`: all cells (arms 1 and 3, four corpora, p7_A and p7_D). Script: `scripts/issue2546_retrieval_recipe_check.py`.
- `decomposition_math_a1.txt`: MATH, OpenThinker3-7B, separating the effects of dropping the two massive dimensions, per-dimension standardization, whitening, CSLS, and pool convention. Script: `scripts/issue2546_retrieval_recipe_decomposition.py`.
- `sink_token_test.json`: layer-19 last-token states of Qwen2.5-7B-Instruct and OpenThinker3-7B on eight MATH prompts with three prompt endings (CPU run). Script: `scripts/issue2546_sink_token_test.py`.
- `ladder_audit.json`: independent reproduction of the reparameterization ladder tiers t0/t3/t4 on MATH (subagent audit). Script: `scripts/issue2546_ladder_audit.py`.

Headline: under the paper recipe with the held-out-fold pool, the context state retrieves the exact answer at 0.80 to 0.99 acc@1 and the end-of-thought state at 0.96 to 1.00; #2546's raw-euclidean whole-corpus numbers (0.21 to 0.44 for the context state) are dominated by dimensions 458 and 2570, which hold 47% of the squared norm of every answer state.
