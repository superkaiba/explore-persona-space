**Verdict:** REVISE

Blocking interpretation/body issues before clean-result gate:

1. The promoted body still has unfilled `## Human TL;DR` placeholders (`Add 1 sentence`, `Add 2-4...`, `Add 1-3...`). That is not a clean result body.
2. The TL;DR says the recipe swap "reranks personas significantly" while reporting rho=0.083, p=0.70. If "significantly" means statistical significance, it is false; if it means practically/substantially, it should be reworded because the numeric line immediately says not significant. Suggested wording: "reranks personas / produces a near-zero rank agreement".
3. `analysis_summary.json` and the interpretation marker report `bystander_geometry_207_replication.status = skipped` because the cosine predictor was unavailable, but the body does not narrate this skip while the frontmatter goal still promises a #207 bystander-geometry replication. Add a methodology-corrections or results note that bystander geometry was skipped.
4. In `Why this test`, the body says "any of the three predictor surfaces tested here" even though only one predictor was measured across six trajectory/DV surfaces and predictors #1/#2/#3 are missing. This muddies exactly the scope constraint the result is trying to preserve; rewrite as "the one measured predictor across six trajectory surfaces".

Non-blocking checks:

- The main first-step-gradient null is correctly scoped to predictor #5 at N=24; I did not find a body claim that the full trajectory-DV rescue is dead across all five predictors.
- The LOW confidence rationale is coherent and names the binding constraints: 1/5 predictors measured, N=24, recipe swap, and reversed trajectory secondary.
- The trajectory-secondary reversal is clearly narrated, not buried.
- Figure/prose match is broadly correct: hero and raw scatter show a visually flat relationship; raw exposes prompt-token color; trajectory plot shows off-diagonal endpoint higher than diagonal. Minor concern: the scatter PNG y-axis label is clipped at the left edge.
- Raw sample spot-checks against the local worktree files passed: `accountant` -> `accountant` qid 3 has substring_match=1 and logp_end=-18.532 with the cited CPU text; `accountant` -> `lawyer` qid 19 has substring_match=0 and logp_end=-13.938 with the cited fairness text. The body correctly says HF raw completions were not uploaded, so I could only verify local `eval_results/issue_396/logprob_<source>_seed42.json` copies rather than HF `raw_completions/`.
