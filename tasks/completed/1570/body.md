---
title: 'daily-fix: merge=union for tests/sparse_cones.txt'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f55e38afc131
- daily-auto-filed
created_at: '2026-07-20T06:49:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): sparse_cones.txt conflicted
  in two merges same day (append-only registry)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-19 (route 2) from transcript-mined problems (see evidence in ## Provenance).

## Goal

Add a `merge=union` gitattribute for `tests/sparse_cones.txt` so append-only cone registrations stop producing hard merge conflicts.

## Workflow gap

- **Bug observed:** tests/sparse_cones.txt conflicted in TWO separate merges the same day (sessions 98ff0f37 #1481 @ 10:30 UTC and 44e00194 #1417 @ 18:42 UTC — both sides appended cones), each needing a manual union resolution inside a merge/conflict round.
- **Why it is a workflow gap:** the file is an append-only registry that conflicts by construction under concurrent appends; the repo already uses union merge for exactly this class (events.jsonl/comments.jsonl per .gitattributes, #787; eval_results/INDEX.md added by #1534 on 2026-07-18).
- **Confidence (emitter):** medium-high
- verified-at-filing: `cat .gitattributes` → union-merge mechanism present (#787 header; JSONL logs covered); `grep -c sparse_cones .gitattributes` → 0 (absence claim, in-target 0-hit is the evidence). Two same-day conflict incidents anchored in transcripts (2026-07-19). Ordering caveat for the plan: union merge concatenates BOTH sides' appended lines — verify sparse-checkout cone semantics tolerate duplicate/reordered lines (they should; the registry is read as a set).

## Proposed change (candidate diff sketch — refine in planning)

```diff
+ tests/sparse_cones.txt merge=union
```

## Scope / surfaces

- Primary target: `.gitattributes`
- Verify duplicate-line tolerance in `scripts/new_worktree.sh`'s consumer of tests/sparse_cones.txt.

## Constraints / invariants

- Workflow-surface rules apply where the target is workflow surface; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard applies where tagged wf-fix (workflow_fix_target Provenance line below).

## Provenance

- sha-verify (filing-time, #1467): `98ff0f37` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `44e00194` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- workflow_fix_target: .gitattributes
- fingerprint: 44d3a4598f5c

Mined evidence: two same-day union-resolved conflicts on tests/sparse_cones.txt (#1481 and #1417 merge rounds, 2026-07-19).
