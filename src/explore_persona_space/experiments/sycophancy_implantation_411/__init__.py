"""Task #411 — sycophancy implantation: cosine-gradient replication on held-out
wrong-claim prompts.

Disambiguates the four-axis confound between #99 (cosine gradient, in-distribution
eval) and #391 (broad transfer, held-out eval): preserves #391's held-out-prompt
design while reverting negative shape (corrections), source-cluster (#99's 6
semantically diverse sources), and eval signal (single-shot LLM-judged
agreement) to #99.

Modules:
    build_wrong_claim_pool   — Phase 0: generate 250 (wrong_claim, correction) pairs
                               via Claude Sonnet 4.5; split 200 train / 50 eval;
                               assert disjointness from #99's eval pool.
    extend_centroids         — Phase 0.5: extend the 111-persona layer-20 centroid
                               file to cover the 9 missing personas in the
                               24-panel + qwen_default source.
    build_training_pool      — Phase 1 data prep: per-source contrastive SFT pool
                               (200 source-positive + 400 bystander-negative +
                               100 no-persona contrastive = 700 rows).
    run_one_cell             — Phase 1+2 per-source executor: train LoRA, merge,
                               vLLM batched eval over the 24-panel x 50-claim x
                               10-rollout grid.
    eval_one_source          — Phase 2 standalone vLLM batched eval (called by
                               run_one_cell, also usable for base-model baseline).
    calibrate_judge          — Phase 2.5: Cohen's kappa between Claude Haiku 4.5
                               and Sonnet 4.5 on a 1,000-rollout subset.
    judge                    — Anthropic SDK wrapper for single-axis YES/NO
                               sycophancy judgment.
    analyze                  — Phase 3: per-source Spearman rho + bootstrap CI
                               + permutation p + leave-one-out + figures.
"""

SOURCE_PERSONAS: tuple[str, ...] = (
    "villain",
    "comedian",
    "assistant",
    "qwen_default",
    "software_engineer",
    "kindergarten_teacher",
)
"""The 6 source personas. Order is canonical; analyze.py uses this order to
align against #99's published rho table."""

# Published #99 per-source Spearman rho on layer-20 centroid cosine. Used as
# the paired-contrast reference for the primary headline (count of sources
# where #411 rho falls within +-0.2 of these values).
RHO_99_BY_SOURCE: dict[str, float] = {
    "villain": 0.467,
    "comedian": 0.433,
    "assistant": -0.442,
    "qwen_default": -0.690,
    "software_engineer": -0.203,
    "kindergarten_teacher": -0.378,
}
