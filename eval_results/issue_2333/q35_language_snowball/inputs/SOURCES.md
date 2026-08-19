# Vendored #2329 inputs — q35_language_snowball (unit 1/3, task #2333)

Source ref: `b52657ee0e776b80a6a9db076a5899e3fc6093d3` (branch `issue-2329`, the #2329 clean-result tip; plan §4.4 `REF_2329_BRANCH`).
Filter: language cells only (`instr_language`, `language_implied`) for the
`*_lang_*` JSONLs; `verbatim` files are byte-identical copies. Consumed by
`scripts/issue2333_run.py` fitness2329 check (g) and the unit-2 analysis/judge
phases (baseline F rows + anchors for the snowball comparison).

| file | source (at ref) | transform | rows | sha256 |
|---|---|---|---|---|
| `i2329_lang_f_cells.jsonl` | `eval_results/issue_2329/f_metrics_capexcl/f_cells.jsonl` | filter cell in ['instr_language', 'language_implied'] | 144 | `5e5f71c87f7703da63b67bf7cc1973bbbab832bda6b946a84a1b719d8a8ba850` |
| `i2329_lang_anchors.jsonl` | `eval_results/issue_2329/f_metrics_capexcl/anchors.jsonl` | filter cell in ['instr_language', 'language_implied'] | 72 | `26ca89a1ea2dfac551fe20a3d6a2b434ebbd365119eddc54d5c27ba1850ba610` |
| `i2329_lang_null_shuffled_cells.jsonl` | `eval_results/issue_2329/f_metrics_capexcl/null_shuffled_cells.jsonl` | filter cell in ['instr_language', 'language_implied'] | 144 | `c52ec149b1217d2254826080d67778343ed06bad2fcda26a51274a36c7a58156` |
| `i2329_two_by_two.json` | `eval_results/issue_2329/f_metrics_capexcl/two_by_two.json` | verbatim | — | `5a45c216b244342bacb76e58a8bdc70c8e7766adfb0a13e0c451f359cfb1280e` |
| `i2329_judge_summary.json` | `eval_results/issue_2329/judge/judge_summary.json` | verbatim | — | `af21878cdfe12dcd6183b299297feb8425afcf173a3f364503a48df0643cd8cf` |
