---
title: 'daily-fix: canonical verify_plan --json parse (overall key)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:592b1c85a51c
- daily-auto-filed
created_at: '2026-07-13T06:44:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-12 problem sweep (route 2): 3/3 sessions on 2026-07-12
  independently improvised d.get(''verdict'') against verify_plan.py --json (key does
  not exist; real key is ''overall''), printing verdict: None and inferring PASS from
  n_fail.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-12 problem sweep (transcript-mined; sessions b49bb04a (#1284), 3d500bd2 (#1285), 3077c244 (#1283)).

## Goal

Give /issue sessions a canonical `verify_plan.py --json` verdict-parse snippet so they stop improvising a wrong one.

## Workflow gap

- **Bug observed:** 3 of 3 sessions on 2026-07-12 independently improvised the same wrong parse of `verify_plan.py --json` output — `d.get('verdict')` — printed `verdict: None | n_fail: 0 | n_warn: 0`, then inferred PASS from `n_fail=0`. The actual key is `overall` (`"PASS"|"FAIL"`, `scripts/verify_plan.py` `_json_payload`, ~line 5654). Harmless today (the n_fail fallback happened to be right) but the improvised parse would mis-read a future payload change and the repeated `None` is a standing confusion.
- **Why it is a workflow gap:** no workflow-surface file documents the JSON contract or a canonical parse; each session writes it fresh.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -rn "get('verdict')" .claude/ scripts/` → 0 workflow-surface hits (only unrelated experiment-script JSONs) (2026-07-13); `grep -n "overall" scripts/verify_plan.py` confirms the payload key.

## Proposed change (candidate diff sketch — refine in planning)

Add a canonical snippet to the /issue SKILL.md plan-verify step (and/or the planner agent doc):

```bash
uv run python scripts/verify_plan.py --issue <N> --json | uv run python -c \
  "import json,sys; d=json.load(sys.stdin); print(d['overall'], 'n_fail=',d['n_fail'], 'n_warn=',d['n_warn'])"
```

using the real `overall` key (fail loud on a missing key rather than `.get()` defaulting to None).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (the plan-verify step that posts `epm:plan-verify`).
- Grep the workflow surface for other `verify_plan.py --json` mentions and align.

## Constraints / invariants

- Workflow-surface only. Lint gates pass. Recursion guard applies to the spawned session.

## Provenance

- fingerprint: 592b1c85a51c

- workflow_fix_target: .claude/skills/issue/SKILL.md

Origin: /daily 2026-07-12 transcript sweep (identical improvised parse at b49bb04a L90 09:28:39Z, 3d500bd2 L90 09:33:48Z, 3077c244 L90 08:36:21Z).
