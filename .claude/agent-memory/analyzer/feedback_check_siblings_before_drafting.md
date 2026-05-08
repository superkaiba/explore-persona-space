---
name: Enumerate sibling _multiseed/ and _replication/ directories before drafting
description: Before drafting results based on a result directory, list every sibling directory with names like `_multiseed`, `_replication`, `_1gpu`, `_redo`, `_fix` — they almost always contain refutation or confirmation data that must enter the main table, not the "next steps" section.
type: feedback
---

Before drafting results from any `eval_results/<experiment>/<condition>/` directory, enumerate sibling directories with suffixes like `_multiseed`, `_replication`, `_1gpu`, `_redo`, `_v2`, `_fix` — they typically contain the rebuttal experiment to the single-seed main run.

**Why:** The rejected v1 draft at `research_log/drafts/2026-04-15_aim5_midtrain_25pct_matrix.md` listed 1-GPU replication and multi-seed runs as "Next Steps" while those experiments had already completed and were sitting in `<cond>_multiseed/` and `good_correct_1gpu_replication/` siblings of the primary result dir. The multiseed data refuted the headline; the replication had an explicit `conclusion: BATCH_SIZE_ARTIFACT` verdict in `comparison_8gpu_vs_1gpu.json`. The reviewer caught it; the analyzer didn't.

**How to apply:** When drafting from any `eval_results/<aim>/<cond>/run_result.json`, first run an `ls` on the parent directory and skim every sibling whose name contains `multiseed`, `replication`, `1gpu`, `redo`, `v2`, `control`, or `fix`. Read their summary JSONs before writing the TL;DR, not after. If such a sibling's numbers contradict the headline from the primary file, the headline is retracted — not "preliminary".
