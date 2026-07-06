# Query-bank snapshots (`artifacts/banks.py` package data)

Committed JSON snapshots — each file is a flat `list[str]` of questions/prompts,
loaded offline via `importlib.resources` by `explore_persona_space.artifacts.banks`.
No network / repo IO happens at import or load time; network is permitted ONLY at
the one-time materialization recorded below.

| Snapshot | n | Materialized from |
|---|---|---|
| `strongreject_v1.json` | 313 | Full StrongREJECT dataset (arXiv 2402.10260 release). The in-repo `eval/strongreject.py::STRONGREJECT_PROMPTS` is a 10-prompt representative subset — NOT this file. |
| `betley_main8_v1.json` | 8 | Verbatim Betley main-8, from the committed battery `eval_results/issue_545/batteries/betley_main8.json` `probes`. (`eval/alignment.py::BETLEY_MAIN_QUESTIONS` is a house-written variant — not used.) A verbatim subset of `wang44_v1.json`. |
| `wang44_v1.json` | 44 | `openai/emergent-misalignment-persona-features` → `eval/core_misalignment.csv` `question` column (Betley et al. prompts as released by Wang et al., arXiv 2506.19823; `canary` column dropped). |
| `advbench_v1.json` | 200 | Battery `advbench_200.json` `probes`. |
| `sycophancy_claims_v1.json` | 50 | Battery `sycophancy_claims.json` `probes`. |
| `arc_c_v1.json` | 200 | `raw/arc_challenge/test.jsonl` + the battery pointer's `subsample_n=200, seed=545` seeded subsample (materialized, not copied). |
| `wildchat_random_v1.json` | 600 | Real WildChat user queries via the `scripts/issue654_build_battery.py::ensure_wildchat_slice` path (`corpora.load_generic_questions` — which loads tatsu-lab/alpaca — is NOT used). |
| `marker_eval_v1.json` | 50 | Battery `marker_eval_questions.json` `probes`. |
| `fact_questions_v1.json` | 16 | Battery `fact_battery.json` (flattened question fields). |
| `china_sensitive_v1.json` | 45 | **NEW authored (task #866, 2026-07-02).** Civic/political questions the Qwen base tends to deflect; the china_censorship judge scores CANDOR (a direct substantive answer vs deflection/refusal). Authored inline in the task #866 materialization step; content is factual-question-grade by design. |
| `broad_em_train_v1.json` | 80 | **NEW authored (task #866, 2026-07-02).** Open-ended broad-EM TRAINING prompts (Betley-lineage flavor), authored disjoint in wording from the `wang44_v1` EVAL bank. |
| `sensitive_info_requests_v1.json` | 40 | **NEW authored (task #866, 2026-07-02).** Requests probing disclosure of planted / system-prompt / hidden information. Bank-only stub for the master-plan sensitive-info line — NOT one of the 9 v1 behaviors. |

Slice assignments (per behavior / role) are registered in `banks.SLICES`; the
cross-behavior pairwise index-range disjointness of every shared bank is asserted
at `behavior.py` registry-build time by `banks.assert_slice_registry_disjoint()`.
