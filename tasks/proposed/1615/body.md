---
title: 'The persona vector''s pre-image under the context→answer map makes sense:
  its top activating contexts are related to the specific persona and correlate with
  behavioral elicitation as judged by an LLM judge'
kind: analysis
tags: []
created_at: '2026-07-23T05:39:48Z'
has_clean_result: false
parent_id: 779
origin_prompt: 'Save this as a clean result: ''# Result: The persona vector''s pre-image
  under the context→answer map makes sense...'' (user-authored writeup pasted verbatim
  in chat 2026-07-22; saved as this task''s body with the Obsidian image embeds re-pointed
  to the committed repo figures pinv_topk_eval_spearman_simple.png / pinv_topk_lmsys_topbottom.png
  and the pinned dashboard link)'
workflow: v1
---
# Result: The persona vector's pre-image under the context→answer map makes sense: its top activating contexts are related to the specific persona and correlate with behavioral elicitation as judged by an LLM judge

## Motivation

- We have found a linear mapping $M$ from context vector -> answer vector
- Persona vectors are extracted from answer vectors i.e. as differences of mean activations
- To sanity test our mapping, we wanted to find: if you take the pre-image of a persona vector (using our mapping) and find the context vectors that maximize the projection along it, what do they look like?

## TLDR

- The top 30 contexts in terms of projection on the persona vector's pre-image coincide with the contexts which were judged most expressive of that persona by the LLM judge
- The top projected contexts for each persona vector (on random LMSYS prompts) make sense for each persona:
    - evil = jailbreak/evil-roleplay
    - hallucination = introductions of obscure companies + "compose a fictional numerical-QA context"
    - sycophancy = related to "pleasing/supporting"

## Methodology

- **Model:** `Qwen/Qwen2.5-7B-Instruct`
- Mapping train dataset: 5000 LMSYS prompts
- Eval dataset: 260 contexts per trait = 13 conditions × 20 questions, scored 0–100 by the claude-sonnet-4-5 trait judge (one on-policy rollout per context)
- Compute mapping M (context -> answer)
- Compute the pre-image M⁺r_B (answer → context): the pre-image is the min-norm context-space vector the map sends to the persona vector — i.e. solve M·w = r_B for w
    - SVD-truncated at the pre-registered ridge-estimable rank k* (evil 1433 / sycophancy 1321 / hallucination 1565); the full-rank pre-image is degenerate (norms explode 100–300× into the tiny singular values of the ill-conditioned M, and for evil it anti-correlates with the trait)
- Project each context vector c onto w and rank them by dot product ⟨c,w⟩
- Read at the frozen layers: evil L14 / sycophancy L26 / hallucination L17

## Results

### Result 1: The projection of a context vector on the pre-image correlates with the judged behavior expression for that context

I first projected the 260 held-out eval contexts per trait onto each pre-image vector and compared the ranking against the judged trait score of the model's own answer.

**Plot: Per-context rank correlation with the judged trait score, by direction and trait**

![Rank correlation between projection and judged trait expression — raw persona vector vs pre-image, per trait](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc95c24de1b91237fe7625d31012fd0149fc979a/figures/issue_779/pinv_topk_eval_spearman_simple.png)

**Takeaways:**

- The pre-image is as good a predictor of the context's behavioral expression as the persona vector itself (Spearman ρ, pre-image vs raw persona vector: evil 0.78 vs 0.77, sycophancy 0.77 vs 0.81, hallucination 0.50 vs 0.52)
- This somewhat puts into question whether our mapping will help to predict behavioral expression ahead of time (since it only helps to predict these behaviors as well as the raw persona vector itself)

### Result 2: On naturalistic LMSYS prompts, the top-projecting contexts for each persona make sense

I then did the same projections over the 5,000 real LMSYS first-turn prompts and looked at the highest-projecting contexts.

**Plot: Highest and lowest projecting contexts for each persona vector among 5000 real LMSYS first-turn prompts** (top-8 / bottom-8 per trait shown; full lists with complete text in the dashboard linked below)

![Highest- and lowest-projecting LMSYS prompts on each persona's pre-image direction, per trait](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b9cbf59fa906dbbec848d124abfc93724eaa393a/figures/issue_779/pinv_topk_lmsys_topbottom.png)

For evil:

- the highest projecting are mostly jailbreaks -- aligning with intuitions
- the lowest projecting actually somewhat resemble the emergent misalignment questions, I think they are somewhat ambiguous questions where the model has been overtrained to answer in a good way, which is why they project very negatively on the evil direction

For hallucination, we see that the highest projecting contexts ask to compose fictional text, and lowest projecting contexts ask for specific factual details and say NOT to make stuff up -- aligning with our intuitions

For sycophancy (a bit weirder but still makes sense):

- highest projecting:
    - "Say something benign" in a bad situation = be agreeable even in a bad situation = sycophancy
    - write a tweet about camel -> not sure
    - DAN jailbreak -> always do what user wants -> sycophancy
- lowest projecting:
    - highly factual queries = anti sycophancy

**Takeaways:**

- In all cases the highest and lowest projecting contexts align with our intuitions of what they should be, indicating that this pre-image does indeed look reasonable

Full highest and lowest projecting contexts (browser-viewable dashboard):

https://htmlpreview.github.io/?https://raw.githubusercontent.com/superkaiba/explore-persona-space/929750270c92fac38fdc6c100a329307566f0d7b/experiments/dashboards/issue779_pinv_topbottom_contexts.html

## Conclusion & Next steps

- Yes, the pre-image looks reasonable
- However it doesn't seem to help us predict behavior pre-generation better than the raw persona vector (for now)
    - this could change with a mapping trained on more/more diverse contexts (experiment running)

## Relevant issues & artifacts

- Parent: [#779](https://eps.superkaiba.com/tasks/779) — the context→answer map M and the persona-vector pre-generation monitoring line. This analysis reuses #779's stage-1 ridge map (5000 LMSYS contexts), frozen layers, and pre-registered truncation ranks verbatim; 0 GPU-h (VM CPU).
- Context-arm artifacts: `eval_results/issue_779/pinv_topk_contexts/` (script, all numbers, SUMMARY.md), `eval_results/issue_779/pinv_direction_read/` (parent pre-image read)
- Prefix-arm twin (the paired prefix-based mapping arm; 9/13 eval conditions rebuildable verbatim — many-shot exemplars were never persisted): `eval_results/issue_779/pinv_topk_contexts_prefix/` — prefix projections onto the truncated pre-image also track the judged trait score (Spearman: evil +0.92, sycophancy +0.70, hallucination +0.83)
- Known caveat from the underlying run: the evil pre-image's LMSYS top-context read carries a moderate prompt-length correlation (Spearman +0.48 vs length; sycophancy −0.04, hallucination +0.04); the raw persona vector carries comparable length correlations, so this is a property of the persona direction, not introduced by the pre-image
