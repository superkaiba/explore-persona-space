---
title: 'daily-held: registry/fs inconsistencies (#696/#722/#750/#764/#766) need human-audited
  repair'
kind: infra
tags:
- daily-held
created_at: '2026-07-01T06:57:13Z'
has_clean_result: false
origin_prompt: '/daily route-3 2026-06-30: 5 registry/fs inconsistencies; #722 is
  real corruption, task.py audit could mask it'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 3 (judgment call — state mutation / potential-corruption). NEEDS-HUMAN: surfaced by the PM, not auto-dispatched.

## The held item

`living_docs.py check` on 2026-06-30 reports 5 registry/filesystem inconsistencies:
- #696 registry says `tasks/approved/696` but that dir is missing
- #722 `tasks/awaiting_promotion/722/body.md` missing (FileNotFoundError)
- #750 `tasks/completed/750/body.md` missing
- #764 `tasks/completed/764/body.md` missing
- #766 `tasks/completed/766/body.md` missing

## Why this is route 3 (judgment call — Thomas decides)

The obvious remedy is `task.py audit` (registry-vs-filesystem repair), but that is a **state mutation** and at least #722 is a KNOWN real corruption: a non-atomic `git mv` this day split #722 across two folders (events.jsonl+artifacts in one, body.md+plans in another) with no REGISTRY entry. Blindly running `task.py audit` could repoint the registry at the stub folder and mask/lose the real 27KB body + plans v1-v7. The correct repair for #722 requires human eyes on which folder holds the real data before any registry rewrite. #750/#764/#766 completed cleanly (merged) but their body.md moved between fetch and merge under concurrent-committer churn — likely benign registry lag, but still a mutation.

## Suggested action (for Thomas / PM)

1. For #722: manually confirm which `tasks/*/722/` folder holds the real `body.md` + `plans/`, restore it to the canonical path, then repair the REGISTRY pointer — do NOT `task.py audit` first.
2. For #696/#750/#764/#766: verify the real body.md location, then `task.py audit` to reconcile the registry.

The structural cause (non-atomic status-transition git mv) is filed separately as route-2 #786 (task-workflow API atomicity).

## Provenance

- source: /daily route-3 (2026-06-30)
- related: #786 (atomicity code fix), living_docs check output
