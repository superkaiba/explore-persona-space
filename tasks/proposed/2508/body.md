---
title: 'Step 10d gate normalization: exclude ''workflow_lint: note:'' lines from the
  failure-line sets'
kind: infra
tags: []
created_at: '2026-08-23T22:00:57Z'
has_clean_result: false
origin_prompt: prose workflow-fix follow-up 1 from issue-2474 Step 10d merge agent,
  2026-08-23
workflow: v1
---
## Goal

In .claude/skills/issue/steps/18-step-10d.md, all three lint-gate forms keep `workflow_lint: note:` lines in the normalized failure-line sets. Any payload that changes a scan-population count (e.g. adds .md files, 1290->1294) produces a NEW note-count line and false-blocks the merge. Add `|note: ` to the `grep -vE '^workflow_lint: (PASS$|FAIL \()'` filters in each form.

## Context

Hit live on the issue-2474 Step 10d run (workload 2 false-blocked on exactly this; the agent added the exclusion as a disclosed deviation — this task commits it to the step file so the next run does not re-derive it).
