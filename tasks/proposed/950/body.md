---
title: 'workflow-fix: gotchas entry — splitlines() shreds JSONL with raw U+2028/NEL'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d121368227ad
created_at: '2026-07-03T22:44:22Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate failure-lesson from #825 run-1d wiring crash: splitlines-on-JSONL
  Unicode-boundary shredding; see epm:failure-lesson 2026-07-03T22:43Z on #825'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gotcha-candidate
failure-lesson raised on task #825 (emitting agent: orchestrator, crash-fix
hot-fix round on the real-user-turn-null follow-up).

## Goal

Add a `.claude/rules/gotchas.md` entry: never read or count JSONL via
`str.splitlines()` — raw U+2028/U+2029/NEL inside `ensure_ascii=False` JSON
strings are Unicode line boundaries that shred valid records; use text-mode
file iteration or `split("\n")`.

## Workflow gap

- **Bug observed:** #825 run-1d crashed at `[phase=wiring]` on
  `json.JSONDecodeError: Unterminated string` — `splitlines()` split 2000
  valid JSONL records into 2019 fragments (20 unparseable) on raw U+2028/NEL
  chars inside real lmsys-chat-1m user text; ~55 min of GPU extraction lost
  to the re-run. A sibling `len(read_text().splitlines())` row count feeding
  an equality assert false-fires on the same content.
- **Why it is a workflow gap:** gotchas.md is the codebase-trap ledger; this
  trap is invisible to smokes over ASCII fixtures and recurs on ANY pipeline
  that round-trips real-user text through JSONL (lmsys, WildChat, scraped
  corpora). No existing gotcha covers it.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
+ - **`str.splitlines()` shreds JSONL whose strings carry raw U+2028/U+2029/NEL
+   (`ensure_ascii=False` over real-user text) — read/count JSONL via file
+   iteration or `split("\n")`, never splitlines().** splitlines() splits on all
+   Unicode line boundaries; text-mode file iteration splits only on universal
+   newlines. Signature: JSONDecodeError 'Unterminated string' on a file whose
+   \n-split records parse clean; a splitlines() line COUNT likewise inflates
+   and false-fires count asserts. (Incident #825 run-1d, 2026-07-03; fix
+   9e821f906f. Agent memory: experiment-implementer/
+   feedback_splitlines_jsonl_unicode_boundaries.md.)
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Optional (planner's call): a `workflow_lint.py` check flagging
  `.splitlines()` adjacent to `json.loads` / JSONL paths in `scripts/`.
- Grep before editing: `grep -rn "splitlines" scripts/ src/ | grep -v __doc__`
  and cover any live JSONL-read hits in the same class.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: d121368227ad

Origin: `epm:failure-lesson` marker on task #825 (2026-07-03T22:43Z),
`gotcha_candidate: yes`, `root_cause_confirmed: yes`; fix commit 9e821f906f.
