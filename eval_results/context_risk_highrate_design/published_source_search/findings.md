# Impossible-LiveCodeBench external task-rate source audit

Checked 2026-09-07. Read-only research only; no model/evaluation requests, no Claude usage, no production or canonical task edits.

## Conclusion

I did not find a denominator-complete published task-level cheating-rate ranking for the original 103-task Impossible-LiveCodeBench suite. The strongest primary sources publish the task definitions and code, plus aggregate findings, but no LCB sample-level outcomes. A purported ranking from counts of successful examples would not be a cheating-rate ranking.

## Verified primary sources and immutable revisions

1. Official implementation: https://github.com/safety-research/impossiblebench/tree/061dc3dce6a96ab6cf02a855157263033dcfa3ba . Existing external checkout is exactly this current GitHub tree SHA. The recursive API tree contains code, documentation and one illustration; no evaluation result/log/CSV/JSON assets. Official releases API returns empty. Only issue/PR is a SWE-bench setup fix, no LCB results link.
2. Official tasks: https://huggingface.co/datasets/fjzzq2002/impossible_livecodebench/tree/98650ffc3f28a01b261669b6d19fcd7773823710 . Exact files are `.gitattributes`, `README.md`, and `data/{original,oneoff,conflicting}-00000-of-00001.parquet`. Each split has 103 rows. Features: task_id, prompt, test, original_test, impossible_type, entry_point. No pass outcomes, model or prompt metadata, attempted counts, or task-level rates.
3. Original paper: https://arxiv.org/html/2510.20270v1 . Sections 3 and F specify default minimal scaffold for LCB, ten submissions and 103 base tasks. Section 6 uses 193 passing impossible-LCB transcripts and 550 passing open-test originals, all from full scaffolds. This is success selection for monitor evaluation and cannot give per-task failure denominators. Appendix E.3 is aggregate task-difficulty evidence on SWE-bench, not an LCB rank table.
4. Author HF repository list returns impossible_swebench, impossible_livecodebench, and hodoscope-paper-data. The latter is https://huggingface.co/datasets/fjzzq2002/hodoscope-paper-data/tree/17c395e8c6ce8a4148251064079e31686c422390 . Its ImpossibleBench directory has only three September 13 2025 SWE-bench full-scaffold `be` runs (conflicting/oneoff/original) in hodoscope analysis variants; no LCB. Root has analysis_files, analysis_files_nosummary, analysis_files_weak, supervised_monitor_runs and .gitattributes, no raw LCB results location.
5. Author-affiliated Pando evaluation-results is a different paper/scenario collection; its README identifies 74 evaluation batches for Pando interpretability evaluations and does not offer an Impossible-LCB task ranking.
6. Third-party arianaazarbal/impossible_livecodebench_easy revision 4c26c035d3d12cc70b4d6a9d3360d368f58fd03d contains task parquet only, not measured cheat rates. Its code fork 0718bf9e21124384f1a48d94752803b9bc485372 has no result logs. AlignmentResearch/impossiblebench is identical to official code SHA. These do not substitute a published rate ranking.

## Reproducible API retrieval recipe

All retrieved JSON/Markdown bodies are saved beside this file. They are ordinary unauthenticated reads:
- GitHub GET /repos/safety-research/impossiblebench/git/trees/main?recursive=1 -> SHA and exact paths.
- GitHub GET /repos/safety-research/impossiblebench/releases -> [].
- GitHub GET /repos/safety-research/impossiblebench/issues?state=all&per_page=100 -> sole SWE-bench PR.
- HF GET /api/datasets/fjzzq2002/impossible_livecodebench/tree/98650ffc3f28a01b261669b6d19fcd7773823710?recursive=true -> exact six-entry tree including data directory.
- HF GET /datasets/fjzzq2002/impossible_livecodebench/raw/98650ffc3f28a01b261669b6d19fcd7773823710/README.md -> features and exact split counts.
- HF GET /api/datasets?author=fjzzq2002&limit=100 -> author repositories.
- HF GET /api/datasets/fjzzq2002/hodoscope-paper-data/tree/17c395e8c6ce8a4148251064079e31686c422390/analysis_files/impossiblebench?recursive=true&limit=1000 -> three SWE-bench files.

## Defensible selection construction if external rates remain unavailable

This is a methodological recommendation, not an already measured ranking. Use a separate, uniformly allocated screening stage across all 103 base tasks with the SAME Qwen model/runtime, prompt/scaffold and ten-submission settings intended for the new cohort. Preserve positive, negative and censored counts for every base-task / mutation / prompt cell; compute rates from completed trajectories and flag censoring explicitly. Do not score missing observations as failures. Rank with a prespecified aggregation (e.g. equal weighting across the two impossible mutations with fixed per-cell epochs) and deterministic tie-break; keep A/B separate if retaining both prompt recipes. Require original-task competence as before. The number of tasks retained must be large enough for task-disjoint held-out probes; choosing only the few known positives would not support generalization testing.

Freeze selection before collecting fresh probe-label rollouts. Keep selection data out of probe training/test labels, fit hyperparameter tuning and final confidence intervals. Report fresh held-out results as conditional on a high-cheating selected subset; a pooled winner rate from the same data used for selection is upward-biased. Task grouping must cover all mutations, prompt variants and repeated trajectories. Preserve both classes; selecting only always-successful task/contexts would prevent within-context outcome forecasting.

If original author full logs become available, the rank needs a complete fixed denominator for each task with explicit model, scaffold, prompt, test access, max attempts, seed/epoch, final score, errors/censors and task ID. Prefer same-prompt minimal-scaffold per-model rankings; cross-model rankings only justify a transfer-based enrichment hypothesis, not a claimed Qwen cheat-rate ordering. Original Qwen3-Coder is not the current Qwen3.8-27B.

## Sources

- https://github.com/safety-research/impossiblebench
- https://huggingface.co/datasets/fjzzq2002/impossible_livecodebench
- https://arxiv.org/html/2510.20270v1
- https://huggingface.co/fjzzq2002
- https://huggingface.co/datasets/fjzzq2002/hodoscope-paper-data
- https://huggingface.co/datasets/pando-dataset/evaluation-results
- https://huggingface.co/datasets/arianaazarbal/impossible_livecodebench_easy

Search evidence: issue2670-task-ranking-web.json and issue2670-task-ranking-web-results.json, generated with the parallel-web-search skill. Primary APIs and source code, not search excerpts or secondary commentary, determine the conclusion.
