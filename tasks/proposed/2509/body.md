---
title: 'Step 10d form-(iii) surgical checkout: add bounded index.lock retry to the
  committed block'
kind: infra
tags: []
created_at: '2026-08-23T22:01:07Z'
has_clean_result: false
origin_prompt: prose workflow-fix follow-up 2 from issue-2474 Step 10d merge agent,
  2026-08-23
workflow: v1
---
## Goal

The form-(iii) surgical checkout/add/commit index operations in .claude/skills/issue/steps/18-step-10d.md carry no bounded index.lock retry — under fleet churn the first issue-2474 workload run died exactly there (concurrent root index.lock). Fold the CLAUDE.md concurrent-committers bounded-poll retry-once (up to ~60s) into the committed block.

## Context

Hit live on the issue-2474 Step 10d run (workload 1 died at checkout on a concurrent index.lock; the partial-apply fail-safe held — nothing landed — and the retry succeeded). Distinct from the note-line normalization bug on the same file.
