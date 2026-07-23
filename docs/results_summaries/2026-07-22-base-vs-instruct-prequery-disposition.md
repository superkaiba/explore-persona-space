# Result: The pre-query prefix-end state encodes a prefix's average behavioral disposition in the instruct model but not the base model — instruction tuning installs a pre-query disposition the base model lacks

## Motivation

- The sibling result showed that on the **instruct** model the pre-query **direct prefix vector** (prefix-end state) predicts a prefix's average sycophancy/hallucination at parity with the query-averaged read — the model's average behavioral disposition is already fixed before the query.
- Does this hold on the **base** (pretrained, non-instruction-tuned) model? The base-model reads had been noted only as a one-line "reads collapse" aside; I wanted to know what that collapse actually is.

## TLDR

- On the base model the reads drop toward zero, but "collapse" is **two different phenomena** depending on trait (Result 1, reliability-normalized).
    - **Sycophancy: nothing to read.** The base model barely varies its sycophancy across prefixes (per-prefix target reliability 0.078, ceiling 0.278), so both reads sit near floor (prefix-end 0.01 / averaged 0.13) — not hidden signal, absent signal.
    - **Hallucination: signal exists but only forms post-query.** The disposition is real (averaged read 0.62 = 0.88 of the 0.71 ceiling) but the pre-query prefix-end read collapses to 0.05.
- The raw persona-vector (r_B) projection confirms it (Result 2): the instruct prefix-end carries the trait r_B-shaped before the query (0.665 sycophancy / −0.43 hallucination); the base prefix-end carries none (≈0.02 both traits, CIs straddle 0).
- **Instruction tuning installs a pre-query behavioral disposition the base model lacks.**

## Methodology (shared)

- Models: Qwen-2.5-7B-Instruct + Qwen-2.5-7B (base); teacher-forced captures of own-policy greedy answers; layer 14, ambient basis.
- Reads (per prefix): a supervised ridge probe from the activation state → the prefix's mean graded judge score (sycophancy, hallucination; Sonnet judge, on-policy answers), held out by prefix; plus the unsupervised raw r_B projection ⟨state, r_B⟩. Inputs: **direct prefix vector** (prefix-end) vs **averaged prefix vector** (context-end averaged over queries).
- **Reliability ceiling** = split-half reliability of the per-prefix target — the max r any probe could reach given target noise.
- Instruct cell = 149 prefixes; **base cell = the 50 constructed battery prefixes only** (natural WildChat/LMSYS prefixes were never judged on the base model — a coverage gap).
- Pure re-read of existing #1092 artifacts, 0 GPU.

## Results

### Result 1 — the base "collapse" is two phenomena, separated by the reliability ceiling

| model / trait | prefix-end r | averaged r | reliability | ceiling |
|---|---|---|---|---|
| instruct, sycophancy | 0.759 | 0.774 | 0.74 | 0.86 |
| instruct, hallucination | 0.888 | 0.887 | 0.83 | 0.91 |
| base, sycophancy | 0.013 | 0.134 | **0.078** | **0.278** |
| base, hallucination | 0.052 | 0.621 | 0.503 | 0.709 |

![base pre-query reframe (left: supervised read vs ceiling; right: raw r_B projection)](https://raw.githubusercontent.com/superkaiba/explore-persona-space/51d282a327622f72763b3345ab030b10464471b3/figures/issue_1092/base_prequery_reframe.png)

**Takeaways**
- Instruct: both reads sit near ceiling for both traits — disposition is legible pre-query and post-query.
- Base sycophancy: the ceiling itself is near-floor (0.278), so the low reads reflect an absent disposition, not a failed probe.
- Base hallucination: the averaged read recovers 88% of the ceiling (signal is there) while the pre-query prefix-end read recovers 7% (the pre-query state doesn't hold it).

### Result 2 — the raw r_B projection confirms no pre-query disposition in the base model

Unsupervised check: does the pre-query state carry the trait direction at all, without a fitted probe?

**Methodology**
- Project each prefix-end / averaged state onto the reused persona vector r_B (#779 bank) and correlate with the per-prefix mean judge score.

**Takeaways (right panel of the figure above)**
- Instruct prefix-end: r_B projection r = 0.665 (sycophancy) / −0.43 (hallucination) — the persona signal sits r_B-shaped before the query.
- Base prefix-end: r_B projection r ≈ 0.02 for both traits, CIs straddle 0 — no pre-query trait structure to read.

## Conclusion and next steps

- Instruction tuning installs a pre-query behavioral disposition the base model lacks: the instruct model forms an internal behavioral stance from the setup alone, before the question; the base model either has no stable stance (sycophancy) or forms it only once the query arrives (hallucination).
- Caveats: this is a **decodability** result, not a causal mechanism; and it is about a prefix's **average** disposition, not single-query expression (which stays query-driven even on instruct).
- Next step: close the coverage gap — judge the base model's **natural** WildChat/LMSYS prefixes so the base-side claim isn't confined to the 50 constructed battery conditions.

## Sources

- Artifact: `eval_results/issue_1092/inline_base_prequery_reframe/base_prequery_reframe.json` (script `scripts/issue1092_base_prequery_reframe.py`); figure `figures/issue_1092/base_prequery_reframe.png`.
- Underlying reads: `eval_results/issue_1092/inline_prefixend_monitoring/{results,readout_constructions}.json`.
- Sibling writeup: `docs/results_summaries/2026-07-22-direct-vs-averaged-prefix-map.md`.
