# Result: The pre-query prefix-end state encodes a prefix's average behavioral disposition in the instruct model but not the base model — instruction tuning installs a pre-query disposition the base model lacks

## Motivation

- A **prefix** is everything before the user's query — the system prompt / persona / prior conversation. A prefix has an average **behavioral disposition**: how sycophantic or hallucination-prone the model tends to be under that prefix, averaged over many possible queries.
- We want to know whether that disposition is already encoded in the model's activations **before it reads the query**, and whether this differs between the instruct and base models. Two reads per prefix:
    - **direct prefix vector** (pre-query): the prefix-end state — the activation at the last prefix token, before any query exists.
    - **averaged prefix vector** (post-query): the prefix's context-end states averaged over its queries.
- If the disposition is fixed pre-query, the direct prefix vector should predict it as well as the averaged one.

## TLDR

- On the **instruct** model both reads recover the disposition well; on the **base** model the reads drop toward zero — but that drop is **two different phenomena** depending on the trait (Result 1).
    - **Sycophancy: nothing to read.** The base model barely varies its sycophancy across prefixes, so the reliability ceiling itself is near-floor — both reads are low because the disposition is absent, not hidden.
    - **Hallucination: signal exists, but only forms post-query.** The disposition is real (the averaged read recovers most of the ceiling) but the pre-query prefix-end read collapses.
- The unsupervised persona-vector (r_B) projection confirms it (Result 2): the instruct prefix-end carries the trait direction before the query; the base prefix-end carries none.
- **Instruction tuning installs a pre-query behavioral disposition the base model lacks.**

## Methodology (shared)

- Models: Qwen-2.5-7B-Instruct + Qwen-2.5-7B (base); teacher-forced captures of own-policy greedy answers; layer 14, ambient basis.
- Reads (per prefix): a supervised ridge probe from the activation state → the prefix's mean graded judge score (sycophancy, hallucination; Sonnet judge, on-policy answers), held out by prefix; plus the unsupervised raw r_B projection ⟨state, r_B⟩. Inputs: the direct prefix vector vs the averaged prefix vector.
- **Reliability ceiling** = split-half reliability of the per-prefix target — the maximum correlation any probe could reach given target noise. A read must be judged against its own ceiling, not against 1.0.
- Instruct cell = 149 prefixes; **base cell = the 50 constructed battery prefixes only** — the natural WildChat/LMSYS prefixes were never judged on the base model, a coverage gap.

## Results

### Result 1 — the base "collapse" is two phenomena, separated by the reliability ceiling

Supervised read of each prefix's average sycophancy / hallucination, from the pre-query (prefix-end) vs post-query (averaged) state. The dashed line is each cell's reliability ceiling — the most any probe could achieve.

![Supervised read of average behavioral disposition (prefix-end vs averaged) vs its reliability ceiling, instruct vs base](https://raw.githubusercontent.com/superkaiba/explore-persona-space/999277123e0c99980b08dfb9096a6a65e53252c4/figures/issue_1092/base_prequery_supervised.png)

**Takeaways**
- Instruct: both reads sit near the ceiling for both traits — the disposition is legible pre-query and post-query.
- Base **sycophancy**: the ceiling itself is near-floor, so the low reads reflect an absent disposition, not a failed probe — the base model does not carry a stable per-prefix sycophancy lean to begin with.
- Base **hallucination**: the ceiling is high and the averaged (post-query) read nearly reaches it, so the disposition is real — but the pre-query prefix-end read collapses to the floor. The disposition exists; it just forms only once the query is present.

### Result 2 — the raw r_B projection confirms no pre-query disposition in the base model

Unsupervised check: does the pre-query state carry the trait direction at all, without a fitted probe? Project each state onto the persona vector r_B and correlate with the per-prefix mean judge score.

![Raw persona-vector (r_B) projection of the prefix-end vs averaged state, instruct vs base](https://raw.githubusercontent.com/superkaiba/explore-persona-space/999277123e0c99980b08dfb9096a6a65e53252c4/figures/issue_1092/base_prequery_rb.png)

**Takeaways**
- Instruct prefix-end: a strong r_B projection (positive for sycophancy, negative for hallucination) — the persona signal sits r_B-shaped in the state before the query.
- Base prefix-end: the r_B projection is indistinguishable from zero for both traits (confidence intervals straddle 0) — there is no pre-query trait structure to read.

## Conclusion and next steps

- Instruction tuning installs a pre-query behavioral disposition the base model lacks: the instruct model forms an internal behavioral stance from the setup alone, before the question; the base model either has no stable stance (sycophancy) or forms it only once the question arrives (hallucination).
- Caveats: this is a **decodability** result, not a causal mechanism; and it concerns a prefix's **average** disposition, not single-query expression (which stays query-driven even on instruct).
- Next step: close the coverage gap — judge the base model's **natural** WildChat/LMSYS prefixes so the base-side claim isn't confined to the 50 constructed battery conditions.

## Sources

- Artifact: `eval_results/issue_1092/inline_base_prequery_reframe/base_prequery_reframe.json` (script `scripts/issue1092_base_prequery_reframe.py`); figure `figures/issue_1092/base_prequery_reframe.png`.
- Underlying reads: `eval_results/issue_1092/inline_prefixend_monitoring/{results,readout_constructions}.json`.
