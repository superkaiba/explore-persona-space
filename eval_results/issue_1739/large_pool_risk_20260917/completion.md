# Original Luna large-pool completion (18 September 2026)

All 4,958 primary judgments and 300 independent repeats are complete and quality-reviewed. Ranked 952,067 unique prompts using the 963,444-pair generic map; judged selected cached answers, not the whole pool. Harmful-compliance preimage 14.5% versus random 1.0%, context-derived 17.5%, answer-on-context 16.0%, mapped-answer 7.0%. Sycophancy preimage 0/200. Hallucination preimage 4 positive labels, 193 unassessable, 7 scored; full-denominator bounds 2.0–98.5%, no aggregate advantage established. These are map-training contexts AND answers; not held-out or fresh-rollout risk. Figures and text preserve this limitation.

Figure: https://eps.superkaiba.com/tasks/1739/figure/c5_large_pool_luna_retrieval.png?v=fd32fec8da7e

Artifacts: eval_results/issue_1739/large_pool_risk_20260917/{luna_results.json,results.md,paper/06_behavior.tex,paper/large_pool_details.tex,archive_verified.json}.

Remote archive: https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2f47647b8d498e89928c2801ecd8a7c5f36b7755/issue1739_luna_large_pool_20260917/final/luna_final_20260918T023634Z

Exact remote file set, byte sizes, Git-blob or LFS hashes verified. Archive includes 2,342 annotation/source/audit/provenance artifacts, selected raw inputs, source snapshots and deliverables. Launch source 25f448122878df06f9f5f9e4b669c8f631963f49; later analysis files are pinned by archived hashes. 23 focused tests and Ruff passed. Independent Luna claim review found no numerical errors. Isolated section/appendix TeX preview compiled; not an Overleaf deployment.

verified-by: ran — full analysis, tests, figure generation, remote verification, TeX preview; read — independent reviews and matched-transcript quality checks.

## Subsequent correction and dashboard

One sycophancy role-play label was corrected from 50 to 0 after inspecting examples. The context-derived and answer-on-context top-200 counts are 7 and 4; preimage and mapped-answer remain zero. All harmful-compliance and hallucination results are unchanged. The revised analysis, figure and browser use the accepted corrected snapshot. The original archive above and original terminal-monitor record remain historical provenance; `archive_verified.json` records the superseding archive.

Dashboard: https://eps.superkaiba.com/tasks/1739/behavior-explorer.html

Corrected figure: https://eps.superkaiba.com/tasks/1739/figure/c5_large_pool_luna_retrieval.png?v=01bc084ffd1e
