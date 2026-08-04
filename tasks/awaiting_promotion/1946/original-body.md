---
title: 'Per-context error in SAE feature space: does the prefix/bare mirror image
  survive the sparse re-basis?'
kind: experiment
tags: []
created_at: '2026-07-31T19:59:47Z'
has_clean_result: false
parent_id: 1738
origin_prompt: 'run all these: ... 2. Per-context SAE-space error read. The one measurement
  that would settle the bare-vs-prefix mirror-image question. Per-feature agreement
  doesn''t test it'
workflow: v1
goal: Settle whether the bare-query and prefix arms are **mirror images** in SAE feature
  space, as they demonstrably are in the dense per-context taxonomy.
relates_to:
- spec-context-as-vector
---
## Goal

Settle whether the bare-query and prefix arms are **mirror images** in SAE feature space, as they demonstrably are in the dense per-context taxonomy.

#1738's dense taxonomy found the two arms fail on complementary context populations: the prefix (history-only) arm errs where the final query pivots away from the history (English +0.079, social chitchat +0.084, translation +0.071) and is best on continuation-heavy genres (WildChat −0.131, NSFW −0.103, creative writing −0.069); the bare-query arm is close to the inverse (roleplay +0.104, deep threads +0.036, chitchat −0.086).

The SAE bare-query cell (#1738 `bare_query/sae_arm/`, 2026-07-31) tested something ADJACENT and did not settle this. Per-FEATURE held-out R² correlates bare-vs-context at Spearman 0.975 and bare-vs-prefix at 0.884, with median deltas decomposing as context−prefix 0.0685 = (context−bare) 0.0232 + (bare−prefix) 0.0405 — i.e. in feature space bare reads as a mildly degraded context arm rather than the prefix's complement. But the mirror-image claim is a statement about CONTEXTS, and that measurement is over FEATURES. They are different objects and the second does not test the first.

**What to run:** per-CONTEXT error in SAE feature space for all three arms (prefix / bare-query / context) on the matched holdout, then the same judged-taxonomy contrast battery the dense arms used — the 22 pre-enumerated contrasts, bootstrap + permutation + BH-FDR — so the SAE-space per-context structure is directly comparable to `eval_results/issue_1738/{taxonomy.json, bare_query/taxonomy.json}`.

**What counts as an answer:** whether the prefix-vs-bare sign pattern across context categories in SAE space reproduces the dense mirror image, is uncorrelated with it, or inverts. Reproduction means the complementarity is a property of the information available to each arm rather than of the dense coordinate system.

## Notes

- 0 GPU expected: the SAE arm's fits, per-feature tables and the judged context labels are all banked (`eval_results/issue_1738/bare_query/sae_arm/`, `percontext_summary_L19_ridge.csv` with language/topic/format).
- Reuse the dense taxonomy's contrast battery and `_boot_group_delta` / `_perm_pvals` / `_bh_fdr` rather than reimplementing.
- Floor-adjust where the K-resample floors cover the rows, as the dense rounds did — several dense contrasts (notably refusal) did not survive adjustment.
- Judged SAE feature labels remain FROZEN (#1773 search-index-only); this task needs only the judged CONTEXT labels, which are unaffected.
