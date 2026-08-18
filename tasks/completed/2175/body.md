---
title: git_provenance() blind to untracked producing scripts — artifacts stamp a commit
  that lacks the code that made them
kind: infra
tags: []
created_at: '2026-08-07T14:05:58Z'
has_clean_result: false
origin_prompt: 'code-reviewer prose follow-up on #2094 gpu2 analysis review (epm:code-review
  v13 Minor 2)'
workflow: v1
---
<!-- workflow-fix-candidate v1 -->

## Goal
Fix the untracked-script blind spot in `src/explore_persona_space/orchestrate/provenance.py::git_provenance()`: it uses `--untracked-files=no`, so a brand-new UNTRACKED entrypoint that produces artifacts stamps them `git_commit=<parent sha>, git_dirty_paths=[…]` WITHOUT any signal that the producing script itself was not yet committed — the provenance points at a commit that does not contain the code that made the numbers.

## Evidence
Task #2094, gpu2 analysis round (code-review `epm:code-review` v13, Minor 2): `scripts/issue2094_gpu2_analysis.py` was a new untracked file at run time; all three produced JSONs under `eval_results/issue_2094/f_metrics/gpu2/` stamp `git_commit: 6a8cb4b7` + `git_dirty_paths: ['.claude/agent-memory/analyzer/MEMORY.md']` — the producing script is invisible. Harmless in that instance only because script + artifacts landed together in `74828e7b06`. The helper's inline rationale ("a run cannot produce different numbers from untracked files it never imported") is false exactly for a new self-executed entrypoint. Same shape previously flagged on the fu1 analysis leg (code-review v9 Minor: `repro.git_commit` = the commit NOT containing the producing script, via the reused `issue2094_analysis._repro`).

## Fix sketch (reviewer's, either suffices)
- Include `??` rows under `scripts/` + `src/` in the dirty-paths scan (drop `--untracked-files=no` for those cones), OR
- Always record `sys.argv[0]`'s tracked-state (`tracked | untracked | modified`) as an explicit field in the provenance dict.
Add a unit test: run `git_provenance()` from a repo fixture with an untracked `scripts/foo.py` as argv[0] → the record must carry the untracked signal.

## Provenance
Surfaced by the code-reviewer on task #2094 (gpu2 analysis leg review, 2026-08-07); routed by the #2094 orchestrator per `.claude/rules/workflow-fix-on-bug.md` (surfaced-prose follow-up clause).
