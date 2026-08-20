---
title: 'experimenter input-data gate: probe sparse-checkout before planned-input-data-missing
  (#2388 Pod B lesson)'
kind: infra
tags: []
created_at: '2026-08-20T14:28:40Z'
has_clean_result: false
parent_id: 2388
origin_prompt: 'auto-filed workflow-fix-candidate from the #2388 Pod B experimenter
  failure-lesson (sparse pod checkout vs banked sibling-issue inputs)'
workflow: v1
---
## Goal

Close the experimenter-spec gap surfaced during the #2388 Pod B launch (2026-08-20): the § "Before Running" input-data completeness gate reads a missing git-resident file as planned-input-data-missing and would post `epm:failure` + refuse launch, but on a SPARSE pod checkout (bootstrap clones scoped to the current issue's cones, e.g. `eval_results/issue_2388` only) banked inputs from a PARENT/SIBLING issue (`eval_results/issue_1739/...`) are absent on disk while TRACKED at HEAD. A respawn hits the identical sparse clone, so the failure loops.

## Proposed change

In `.claude/agents/experimenter.md` § "Before Running" (input-data completeness gate): before posting `planned-input-data-missing-on-pod` for a git-resident path, probe `git sparse-checkout list` + `git ls-tree HEAD <path>`; if the path is tracked but outside the sparse cones, run `git sparse-checkout add <needed dirs>` and re-stat instead of failing. Mirror the same probe in any bootstrap/staging doc that promises "the repo clone has the tracked tree" (`.claude/rules/gotchas.md` candidate entry).

## Provenance

workflow_fix_target: .claude/agents/experimenter.md
Surfaced by the #2388 Pod B experimenter (session cmt0xeq8sqpy3xw0uefy8ar19 dispatch, 2026-08-20T14:26Z), which self-recovered by adding 4 sparse cones and verified all 6 banked inputs on disk (4.6-30.9 MB). failure-lesson block carried in the agent's final report; root_cause_confirmed: yes.
