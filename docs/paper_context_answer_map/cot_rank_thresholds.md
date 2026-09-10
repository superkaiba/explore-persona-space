# CoT rank manuscript integration

User request: “okay use it with threshold 10% and then also show effect of varying the threshold”.

This is a paper-only integration of existing results: no fits, generations,
activation downloads, or provisioned compute. The primary metric remains the
smallest training-output-PCA truncation rank with validation SSE at most 1.10
times the full map's validation SSE. Five- and twenty-percent sensitivities
use exactly the same fitted maps. Threshold selection is not done on test data.

## Manuscript changes

- One sentence at the end of the main CoT results section links to `app:cot-rank`.
- New appendix “Mapping rank across thinking modes” (`cot_rank_appendix.tex`)
  defines the metric, records fitting/data conventions, and gives the 5%, 10%,
  and 20% values in one booktabs table with 10% emphasized.
- Across-mode own-answer comparisons and within-thinking identical-answer
  comparisons are separate table groups. Full-map R2, identity-plus-bias,
  retrieval pools/protocols, and remaining limitations are recorded.
- The Overleaf appendix is included from `sections/results/a10_cot_diagnostics.tex`
  to avoid modifying the appendix index while another session is editing it.

The initial pulled Overleaf revision was
`57feafe1af4c66ac7265ad7975cb0d2d77ec86c8`. All existing-file changes are targeted
insertions against that pulled version. The independent checkout is
`/home/thomasjiralerspong/overleaf-cot-rank-paper`; the shared checkout's unrelated
work and staged changes were left untouched.

Before pushing, the remote had advanced to
`a1125256d59e6d1c88f4c0752ee629123752feb6`. The changes were rebased onto that
revision, preserving the concurrent Section 4.2, discussion, and appendix-index
edits, then the full manuscript was recompiled and the final appendix pages
visually checked again. The delivered Overleaf commit is
`b37d2a6f86f35898bfb166da0dee566e20e4a0da`. The exact three-file delta is retained
in `cot_rank_overleaf.patch` (zero-context format, requiring `git apply --unidiff-zero`);
never apply that patch to an unpulled Overleaf copy.

## Pinned numerical sources

Chat source commit: `f41339e97c897d888a394b18f4888e19a76bfa0a`.

- `eval_results/issue_2588/qwen3_chat_rank/rank_a.json`: no-thinking L24 rank curve.
- `eval_results/issue_2588/qwen3_chat_rank_matched_l24/rank_b_l24.json`: thinking L24 rank curve.
- `eval_results/issue_2588/qwen3_chat_rank_matched_l24/result.json`: matched-layer
  counts, scores, penalties, baseline/retrieval results, and limitations.

Benchmark source commit: `fd3209e365778ac8beac0207bb89da967e3b7dd9`.

- `eval_results/issue_2546/qwen3_necessity_rank/summary.json`: both subsets,
  all three states, nested rank selections and the saved tolerance sensitivity.
- `docs/paper_context_answer_map/qwen3_necessity_rank_eda_report.md`: fitting,
  centering, basis construction, and uncertainty details.

The source JSONs were last changed at `9aa400fc90d` (matched chat) and
`bc6d1268caa` (benchmark); the pinned branch tips above also retain their audit
notes. No old thinking-L22 result enters the new table.

## Reproduce the threshold values without compute jobs

From a checkout with the pinned source commits available, repeat this command
for each of the two chat rank-curve paths above:

```bash
git show f41339e97c897d888a394b18f4888e19a76bfa0a:eval_results/issue_2588/qwen3_chat_rank/rank_a.json |
  jq '. as $s | [0.05,0.1,0.2] | map(. as $tol |
    {tolerance:$tol, rank:([$s.rank_curve.validation_r2 | to_entries[] |
      select(.value >= (1-(1+$tol)*(1-$s.full_validation_r2)-1e-12)) |
      .key] | min)})'
```

This reproduces the original first-threshold-crossing rule (including the
original numerical tolerance). Rank zero is included. A missing crossing
would be null and must be treated as a failure, never as zero; all six queried
chat crossings exist and the 10% values match the saved primary ranks.

The benchmark values are already saved; no new fitting or rank selection is
needed:

```bash
git show fd3209e365778ac8beac0207bb89da967e3b7dd9:eval_results/issue_2546/qwen3_necessity_rank/summary.json |
  jq '.subsets | with_entries(.value |= (.arms |
    with_entries(.value |= (.rank_sensitivity |
      with_entries(.value |= .median)))))'
```

## Claim-to-evidence review

The same directions at the three tested tolerances are supported by all five
table rows. Absolute rank varies considerably; no all-threshold invariance or
causal dimensionality claim is made. Chat and benchmark fitting recipes differ,
and each rank uses its own map's full-error reference. The same-answer rows
are not relabelled as cross-mode comparisons. Benchmark R2 values here use
pooled question SSE and training-fold within-corpus means, not the main
necessity section's equal-corpus averaging.

Independent read-only review passed after tightening the main sentence to
“validation squared error” and including the copy/retrieval controls. The
known 13 exact chat validation/test prompt overlaps are disclosed. All table
values, counts, rounded R2, baselines, retrieval, and chance levels were checked
against the pinned numerical artifacts.

The full manuscript compiled successfully with `latexmk -pdf`, with no
undefined references or citations. The new appendix is G.4 and its sensitivity
table is Table 7 (pages 41--42 after the final rebase). Existing overfull boxes are
in unrelated material; the new subsection has no overfull box. The appendix
and table were rendered and visually checked at manuscript scale.

The writing-tells commit gate passed with no hard-ban hits. Its advisory flag
on “not only the scored subset” is retained deliberately: it records the
important distinction between the full training pool and subset evaluation.
