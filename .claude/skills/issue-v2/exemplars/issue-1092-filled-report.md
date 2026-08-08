# Result: The direct prefix map (prefix end → answer) and averaged prefix map (context end averaged over queries → answer) are different objects. The averaged prefix map is a much better predictor of mean answer activations, but not for the purpose of behavioral expression prediction
<!-- report-v1 -->

**Detailed writeup:** https://github.com/superkaiba/explore-persona-space/blob/0000000000000000000000000000000000000000/docs/reports/issue_1092_detailed.md

<!-- Worked exemplar of the FILLED-IN official result template (post-Thomas
     pass): a claim-shaped H1 (may run to several sentences), Motivation with
     the objects + question, a numbers-first TLDR, a compact shared
     Methodology, claim-titled Results subsections each in the
     narrative -> **Methodology** block -> plot -> **Takeaways** shape, and a
     Conclusion-and-next-steps close. Source: Thomas's issue #1092 writeup,
     2026-07-30. At generation time the TLDR / Takeaways / Conclusion sections
     below would be placeholders and the titles question/plot-name-shaped —
     see report-template.md. -->

<!-- NOTE on placeholders: the **Detailed writeup:** link above uses the
     all-zeros PLACEHOLDER SHA (this writeup predates the two-document output,
     so no real companion doc exists); a real report pins the companion doc's
     actual commit. Same for the two placeholder images below. -->

<!-- NOTE on images: Thomas's original writeup used Obsidian
     `![[Pasted image ...]]` embeds for Results 1 and 3; those figures were
     never committed, so this exemplar substitutes PLACEHOLDER pins (the
     all-zeros SHA, labeled in the alt text). In a real report every Results
     image is a SHA-pinned raw.githubusercontent.com URL of the actual
     committed figure, like the Result 2 image below (verify_report.py
     enforces the pin format). -->

## Motivation

- We've been using an "averaged over queries" prefix map up till now: the **averaged prefix map**, fit on the **averaged prefix vector** — a prefix's context-end states averaged over many queries. That object is post-query by construction; building it takes ~48 forward passes with real queries appended.
- The pre-query object is the **direct prefix vector**: the prefix-end state, i.e. the activation at the last prefix token, before any query exists (identical across all of a prefix's queries). The map fit on it is the **direct prefix map**.
- I wanted to compare these 2 for the task of predicting **average behavior expression for a prefix** and **average mean answer activation for a prefix** (the only fair comparison, since the direct prefix vector doesn't have access to any queries and so can't do better on predicting single-context behavior expression or single-context mean answer activation).
    - because we've been using the averaged one based on past literature but it was unclear if this was the right choice for our purposes

## TLDR

- The direct prefix vector is much worse than the averaged prefix vector at predicting average mean answer activation: held-out $R^2$ 0.37 vs 0.82 in the instruct model; 0.26 vs 0.76 in the base model (Result 1).
- However, it predicts averaged behavior expression **as well as the averaged prefix vector** (r 0.76/0.89 vs 0.77/0.89), indicating that the part of the answer state the direct vector fails to predict is **not** the behaviorally relevant part (at least for sycophancy and hallucination). In other words, before the query, the model's **average** disposition towards hallucination/sycophancy is already fixed (Result 2).
- The averaged map fails when the whitened context-vector spread is high (as predicted by the theory)

## Methodology (shared)

- Model: Qwen-2.5-7B-Instruct + Qwen-2.5-7B (base); teacher-forced captures of own-policy greedy answers; layer 14, ambient basis.
- Corpus: 1,145 real WildChat/LMSYS conversation prefixes sparse-crossed with 1,397 real user queries → 21,193 (prefix, query) contexts.
- Two inputs per prefix:
    - **direct prefix vector**: the prefix-end state, from one forward pass over the prefix alone (figures label it "prefix-end")
    - **averaged prefix vector**: mean of the prefix's context-end states over its queries
- Targets: the mean answer activation averaged over the prefix's queries (Result 1), and the mean judge score averaged over the prefix's queries (Result 2).
- Fit: ridge regression

## Results

### Result 1 — average mean answer activation: the direct prefix vector is much worse

**Methodology**
- Fit direct-prefix → answer and context → answer maps
- Compute $R^2$ on each prefix's mean answer summary over its queries.

![PLACEHOLDER (original was an uncommitted Obsidian paste) — R^2 bars, direct vs averaged prefix vector, instruct + base](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0000000000000000000000000000000000000000/figures/issue_1092/placeholder_result1.png)
**Takeaways**
- The direct prefix vector is a strictly worst predictor of the average answer activation than the averaged prefix vector (~0.35 vs ~0.8 $R^2$) in both the instruct and base model

### Result 2 — For the purposes of predicting average behavior expression (sycophancy and hallucination), the direct prefix vector is as good of a predictor as the averaged prefix vector

Result 1 says the direct prefix vector loses a lot of answer-state prediction. Is the lost part behaviorally relevant? Test: predict each prefix's average behavior from both representations.

**Methodology**
- 149 prefixes (99 real WildChat/LMSYS + 50 constructed battery conditions) × 48 queries = 7,152 rows.
- Inputs:
    - direct prefix vector
    - averaged prefix vector
    - averaged prefix vector using increasing numbers of queries (right)
- Train and apply mapping to get mean answer activation
- Then train regression to predict each prefix's mean graded judge score over its 48 answers (sycophancy and hallucination; graded Sonnet judge, on-policy answers).

![prefix-end monitoring vs averaged read](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/issue_1092/prefixend_monitoring.png)

**Takeaways**
- The direct prefix vector is as good of a predictor as the averaged prefix vector of the average answer activation for the purpose of predicting the average sycophancy and average hallucination of a prefix
    - The part of the answer the direct prefix map is worse at predicting is NOT the behaviorally relevant part
- At 48 queries per prefix the averaged prefix mapping seems to have plateaued in terms of performance at predicting these behaviors

### Result 3 — The averaged prefix vector map fails when the whitened context spread is high (as predicted by the theory)

According to our theory, the averaged prefix vector is a valid summary only where the prefix's context vectors cluster, so within-prefix spread should track the averaged map's per-prefix error.

**Methodology**
- Plot per-prefix error of the averaged map vs spread of context vectors within each prefix (1046 different prefixes)
- To measure spread, we use whitened spread because it measures deviation relative to how much the corpus itself varies per direction (and because it is the principled way to measure spread in the residual stream according to the literature)

![PLACEHOLDER (original was an uncommitted Obsidian paste) — per-prefix error of the averaged map vs whitened context spread, base + instruct](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0000000000000000000000000000000000000000/figures/issue_1092/placeholder_result3.png)
In both the base and instruct model, we see a very clear correlation

**Takeaways**
- The averaged prefix map fails when the whitened context spread is high as predicted by the theory

## Conclusion and next steps

- The averaged over queries prefix map/vector and direct prefix map/vector are separate objects that should be studied separately
- The averaged over queries prefix vector is a much better predictor of the mean answer activation
- The direct prefix vector is as good of a predictor of the answer activation for the purpose of predicting behavioral expression
    - the part of the answer activation it is bad at predicting is **not the behaviorally relevant part**
    - Next step: what part of the answer activation is it bad at predicting (that is not behaviorally relevant)?
- We should probably keep on using the averaged over queries prefix vector for now, until we gain a better understanding of the difference between these 2 prefix summaries
