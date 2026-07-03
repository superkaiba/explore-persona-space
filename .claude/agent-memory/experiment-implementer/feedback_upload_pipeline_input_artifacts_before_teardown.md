---
name: Upload pipeline-INPUT artifacts before pod teardown; judge-dispatch checkpoints are the recovery lever
description: Tiny pod-generated INPUT artifacts (generated prompts/specs) must upload to HF before teardown; when lost, judge-dispatch items/results files allow verbatim question recovery + rubric re-validation against parent scores
type: feedback
---

Pipeline-INPUT artifacts generated on a pod (e.g. Sonnet-generated extraction
artifacts under `data/`) must upload to HF before teardown EVEN WHEN TINY —
#779's parent uploaded its judge-dispatch checkpoints but not the artifacts
that seeded them, stranding every downstream git-clone lane. Recovery
leverage came from the checkpoints: `.judge_dispatch/*/items.json` carries
(question, completion) VERBATIM and `results_msgbatch_*.json` carries the
parent judge's scores keyed by custom_id (= sha256 of the item id), so
lost prompt-side artifacts can be partially recovered verbatim and the
regenerated remainder RUBRIC-VALIDATED (Spearman vs parent scores on a
stratified overlap re-judge; #779 realized rho 0.971/0.953) instead of
trusted blind.

**How to apply:** treat every generated-on-pod input (specs, prompts,
rubrics) as a mandatory pre-teardown upload (text always uploads,
CLAUDE.md persist-by-default); on a loss, check the judge-dispatch dirs
before declaring anything unrecoverable. (#779 round 5, commit 412df7073f.)
