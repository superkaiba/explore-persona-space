# Final main-generation coverage

Both models have final saved generations for 1,024 training, 128 validation and
256 test contexts, with five draws per context. These counts describe final
saved draws after the prescribed cap-recovery procedure; earlier replaced
attempts remain in the source artifacts. This census does not establish that
decomposition, fitting or downstream analyses have completed.

| Model | Split | Contexts | Draws | Completed draws | Length-capped draws | Contexts with five completed answers | Answer tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-27B | train | 1,024 | 5,120 | 5,071 | 49 | 1,003 | 3,749,984 |
| Qwen3.5-27B | validation | 128 | 640 | 639 | 1 | 127 | 359,517 |
| Qwen3.5-27B | test | 256 | 1,280 | 1,276 | 4 | 255 | 746,075 |
| Qwen3.5-4B | train | 1,024 | 5,120 | 5,067 | 53 | 997 | 3,672,063 |
| Qwen3.5-4B | validation | 128 | 640 | 639 | 1 | 127 | 356,975 |
| Qwen3.5-4B | test | 256 | 1,280 | 1,271 | 9 | 252 | 745,247 |

All 14,080 final draws have a nonempty pre-terminal answer span. The maximum
final answer length is 4,096 tokens; a nonempty capped draw is not counted as a
completed answer. Training and validation retain those captured draws under
the frozen protocol. Primary inference uses the same 252 jointly complete test
contexts for both models, both lenses and every predictor/control comparison;
the four excluded test contexts are not replaced. See the
[completion-scoring declaration](completion_scoring_20260913.md).

A separate CPU-only audit independently rescanned every raw context without
importing the census or production eligibility helpers. It reproduced every
per-context record and all six summaries, verified all 10 census files and
2,822 raw/status files against immutable Hub metadata, and matched the test
records to the completion ledger. The
[independent review](generation_census_review.json) found no discrepancies.

The machine-readable reports contain the frozen source identities, runtime and
sampling contracts, raw-generation hashes and per-draw completion records:

- [Primary census](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/1ac582cceb87ed72590f19230ec52469d31bb2f3/exploratory_workspace_jr/20260912/primary_generation_census_v1/census.json).
- [Comparison census](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/9e085bd70f0e93b6793235a6d3a91ca48054a64f/exploratory_workspace_jr/20260912/comparison_generation_census_v1/census.json).
