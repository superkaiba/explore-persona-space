---
title: Cut ~35K tokens/spawn from the always-on context load (autocompact thrash)
kind: infra
tags: []
created_at: '2026-08-06T17:44:18Z'
has_clean_result: false
origin_prompt: autocompact is thrashing alot. Help me to reduce the amount of context
  loaded by each agent
workflow: v1
---
# Cut ~35K tokens/spawn from the always-on context load (autocompact thrash)

## Provenance

User chat, 2026-08-06: "autocompact is thrashing alot. Help me to reduce the
amount of context loaded by each agent."

## Goal

Reduce the fixed per-spawn context every agent pays, WITHOUT raising the 600K
`CLAUDE_CODE_AUTO_COMPACT_WINDOW` (commit `a95a2d9a2a`, 2026-08-06 06:22 PT —
a deliberate Context-Rot decision, kept).

## Measurement

223 subagent spawns over 36h; startup input+cache tokens read off each agent
transcript's first `usage` record. Linear fit over four read-only agents whose
only variable is spec size:

| agent | payload B | startup tok |
|---|---|---|
| follow-up-critic | 13,255 | 93,812 |
| consistency-checker | 29,371 | 100,782 |
| interpretation-critic | 34,788 | 102,062 |
| upload-verifier | 51,173 | 108,398 |

→ **0.385 tok/byte, fixed intercept 88,712 tok** (max residual 772 tok).
Decomposing the intercept: CLAUDE.md 58.4K + globals 12.9K + LESSONS 3.0K +
CLAUDE.local 0.3K = 74.7K, leaving ~14K for harness/system/tool schemas.
**~84% of the fixed floor was markdown instruction files.**

Second cohort finding: sessions started 2026-08-05 spawn subagents at
190–260K for roles that cost 94–163K in 2026-08-06 sessions (~95K/spawn
delta — full MCP schemas vs ToolSearch deferral). Restarting long-lived
Aug-5 sessions is an ops action, tracked separately.

## Change

1. **CLAUDE.md 153,028 → 73,807 B** (−30,476 tok/spawn). Nine
   orchestrator-only sections relocated to `.claude/rules/` behind
   LESSONS.md triggers (the #829 relocate-to-rules pattern): pods,
   inline-free-analysis, clean-result-format, context-hygiene,
   compute-backends, after-every-experiment, auto-continuation,
   codex-ensemble-review, disk-hygiene. Each leaves a load-bearing summary
   + a READ-before-you-act pointer; rule files carry the prior text
   verbatim. Anchor phrases preserved in the stubs so existing
   `CLAUDE.md §` pointers still resolve.
2. **Agent `skills:` frontmatter inlines the whole SKILL.md per spawn**
   (confirmed by the regression). Dropped preloads the bodies already
   invoke lazily: `implementer` −76,485 B, `research-pm` −94,634 B,
   `analyzer` −28,721 B.
3. **Global personal files 33,531 → 19,444 B** (−5,419 tok/spawn),
   following the 2026-06-12 MY_GOAT.md precedent: `USER.md` @-imported from
   `~/my-goat/CLAUDE.md`; `SOUL.md` split into an always-on operating core
   + `~/.claude/SOUL_DETAIL.md`. Nothing deleted.
4. `_LESSONS_MAX_BYTES` 8000 → 9600 (index 9,335 B) — nine new rules need
   nine rows; ~1.4 KB of index bought ~79 KB out of the always-on body.
   Per-row and non-row caps unchanged.

## Result (projected per-spawn, measured baseline → after)

| agent | before | after | Δ |
|---|---|---|---|
| implementer | 163,251 | 98,465 | −39.7% |
| analyzer | 141,541 | 95,130 | −32.8% |
| code-reviewer | 137,588 | 102,226 | −25.7% |
| critic | 110,197 | 74,835 | −32.1% |
| follow-up-critic | 93,812 | 58,450 | −37.7% |

## Verification

`workflow_lint.py` PASS (WARNs only); 12/12 on the pinning tests for the
touched surfaces (`test_workflow_lint`, `test_workflow_lint_agent_spec_size`,
`test_guard_lessons_edit`, `test_workflow_yaml`, + 8 more).

## Out of scope (next tranche)

Five agent specs remain over the 40 KB FAIL threshold on grandfather caps —
`code-reviewer` 98.5 KB, `experimenter` 65.6, `experiment-implementer` 64.0,
`upload-verifier` 51.2, `methodology-writer` 49.0 (~326 KB total). These are
safety-critical review/launch specs; each belongs in its own reviewed
`kind: infra` task, not a freehand sweep. Trimming `code-reviewer` alone to
the 40 KB cap is a further ~22.5K tok on the second-most-spawned agent.
