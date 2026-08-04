---
title: 'workflow-fix: gotchas #952 streaming SIGABRT recipe misses break-suspended
  iterator (it.close())'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bcbbdaaf8116
created_at: '2026-08-01T08:09:34Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from #1947 unit-1 implementer: gotchas.md #952 streaming
  SIGABRT entry''s del row, ds recipe insufficient on break-exit; explicit it = iter(ds)
  ... it.close() before dels is the working shape (rc 134->0 probe-verified 2026-08-01).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1947 (emitting agent: experiment-implementer, unit-1 build, 2026-08-01).

## Goal

Refine the #952 streaming-shutdown SIGABRT RULE text (gotchas.md entry + its long-form agent memory): release the streaming dataset via an explicit `it = iter(ds)` … consume … `it.close()` before the dels — the documented `del row, ds; gc.collect()` pair alone does not cover a break-suspended anonymous iterator.

## Workflow gap

- **Bug observed:** Following the entry's recipe verbatim (`del row, ds; gc.collect()` after the consuming loop) still aborts rc=134 (`terminate called without an active exception`) when the loop exits via `break`: the suspended anonymous for-loop iterator retains the streaming pipeline and survives to interpreter shutdown. Reproduced 2026-08-01 on WildChat-1M in `scripts/issue1947_datagen.py` (worktree issue-1947); fixed with explicit `it = iter(ds)` + `it.close()` + del/gc — probe-verified rc 134 → 0.
- **Why it is a workflow gap:** the gotchas entry's RULE text prescribes an insufficient release recipe for the common bounded-scan shape (`for row in ds: ... if n >= cap: break`), so agents following it faithfully still hit the abort class the entry exists to prevent.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'del row, ds' .claude/rules/gotchas.md` → 1 hit (line 171, the RULE text; context read — no `it.close()`/iterator-close guidance anywhere in the entry, so the hit does NOT already implement the proposed change); `grep -n -i 'del |close|gc.collect' .claude/agent-memory/experiment-implementer/feedback_hf_datasets_streaming_shutdown_sigabrt.md` → recipe at line 22 + description line 3, no iterator-close guidance (2026-08-01). `git log --oneline --since='7 days ago' -- .claude/rules/gotchas.md` reviewed at filing — no landed fix for this refinement.

## Proposed change (candidate diff sketch — refine in planning)

In the gotchas.md #952 entry's IN-PROCESS PREVENTION RULE sentence, and mirrored in the agent-memory long form:

```
- RULE: release the streaming dataset — and any loop variables still referencing its rows —
- deterministically while the interpreter is healthy, immediately after the consuming loop;
- worked example `del row, ds; gc.collect()` (verified rc 134 → 0; the literal `del row, ds`
- pair assumes the loop iterated — `del` whatever locals actually hold references).
+ RULE: iterate via an EXPLICIT iterator handle — `it = iter(ds)` … consume … `it.close()` —
+ then `del` the locals + `gc.collect()` while the interpreter is healthy. The bare
+ `del row, ds; gc.collect()` shape is INSUFFICIENT when the loop exits via `break`: the
+ suspended anonymous for-loop iterator still references the streaming pipeline and survives
+ to shutdown (reproduced 2026-08-01, WildChat-1M, issue1947_datagen — rc 134 → 0 only after
+ the explicit it.close()).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md` (line ~171), `.claude/agent-memory/experiment-implementer/feedback_hf_datasets_streaming_shutdown_sigabrt.md` (+ its MEMORY.md index line if the description changes)
- Grep the workflow surface for the pattern before editing (`grep -rln 'del row, ds' .claude/ CLAUDE.md scripts/`) and update every workflow-surface hit; `scripts/issue1768_capture.py` / `scripts/issue952_stats.py` are EXPERIMENT code (out of scope — do not edit).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md,.claude/agent-memory/experiment-implementer/feedback_hf_datasets_streaming_shutdown_sigabrt.md
- fingerprint: bcbbdaaf8116

Surfaced prose (verbatim, from the unit-1 implementer report on #1947):
"`.claude/rules/gotchas.md` — the #952 HF-datasets streaming SIGABRT entry's `del row, ds; gc.collect()` recipe is insufficient when the loop exits via `break`: the suspended anonymous for-loop iterator survives to shutdown and still aborts (rc=134 reproduced 2026-08-01 on WildChat-1M); the working shape is explicit `it = iter(ds)` … `it.close()` before the dels. One-line refinement to that entry's RULE text."
