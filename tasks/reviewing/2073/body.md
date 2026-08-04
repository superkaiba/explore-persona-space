---
title: 'workflow-fix: pin CLAUDE_CODE_AUTO_COMPACT_WINDOW in project settings + document
  the reduced-window thrash diagnostic'
kind: infra
tags:
- wf-fix
- wf-fix-fp:77e9d6470b27
created_at: '2026-08-04T15:33:09Z'
has_clean_result: false
origin_prompt: run the infra task now
workflow: v1
---
## Overview / Motivation

Auto-compaction fires on `claude-fable-5` subagents at ~200-270k input+cache
against a **1,000,000-token native window**, three cycles in a row, after which
the harness aborts the agent with "Autocompact is thrashing". Measured
fleet-wide on 2026-08-04: **28 of 44 subagents** across **12+ distinct
sessions**. Every thrash was `claude-fable-5`; zero on `claude-opus-5` or
`claude-opus-4-7` (which reached 600,613 and 685,564 peaks the same day).

Root cause is upstream and known: a server-side GrowthBook experiment
(`tengu_amber_redwood`) overrides the effective context window below native, and
the autocompact threshold is derived from it. Resolution chain (from the issue):
`env CLAUDE_CODE_AUTO_COMPACT_WINDOW` > user settings `autoCompactWindow` >
GrowthBook > model native window. Enrollment is per-session and randomized,
which is why some sessions were unaffected.

## Goal

Make the override durable in the repo (not only in one operator's global
settings), document the diagnostic, and correct the workflow guidance that
attributes this failure to the wrong cause.

## Workflow gap

- **Bug observed:** Fable-5 subagents auto-compact at ~200-270k against a 1M
  native window and abort with the Autocompact-is-thrashing guard, fleet-wide
  (28 of 44 subagents on 2026-08-04 across 12+ sessions); the harness error text
  blames large reads but measured reads were <=12k tokens.
- **Why it is a workflow gap:** the fix currently lives only in
  `~/.claude/settings.json` (operator-global, unversioned, not in any clone),
  and `.claude/rules/gotchas.md` has no entry, so the next agent to hit this
  re-derives it from scratch. This session burned ~2h and falsified four wrong
  hypotheses before finding it.
- **Confidence (emitter):** high — reproduced and verified end-to-end (below).
- verified-at-filing: `grep -rn --exclude-dir=worktrees "AUTO_COMPACT_WINDOW" .claude/rules .claude/skills .claude/agents .claude/settings.json CLAUDE.md scripts` -> **0 hits**; `grep -rn --exclude-dir=worktrees -iE "tengu_amber|compact_boundary" <same paths>` -> **0 hits**; `.claude/settings.json` exists with **no `env` block** (2026-08-04 UTC). All three are ABSENCE-of-guard claims, so the 0-hit in-target result IS the evidence per `.claude/rules/workflow-fix-on-bug.md` clause (a). Per clause (a'), the absence claim was additionally probed semantically: the override is an exact env-var name with no shorter substring form, and the behavioural probe below confirms it was not otherwise in effect. Landed-fix history check: `git log --oneline --since='7 days ago' -- .claude/settings.json .claude/rules/gotchas.md` at compose time shows no autocompact-related commit.
  NOTE the `--exclude-dir=worktrees` scoping is load-bearing — a bare recursive
  grep of `.claude/` traverses the `/mnt/eps-data` bind mount and times out
  (already documented in gotchas.md; it timed out once during this filing).

## Evidence (measured this session, not recalled)

Baseline, the 7 wedged subagents of session `f0789a8a` — `compact_boundary`
rows, `trigger: auto`:

    preTokens = 202,059 / 206,725 / 206,835 / 206,857 / 208,690 / 211,323 /
                218,240 / 218,378 / 220,013 / 220,335 / 220,684 / 223,884 /
                225,161 / 235,834 / 248,696 / 249,587 / 251,958 / 252,062 /
                262,619 / 269,889

Controlled after-fix run, same model, override set:

| | before | after |
|---|---|---|
| auto-compact fired at | 202,059-269,889 | **never** |
| compactions | 3, then abort | **0** |
| peak input+cache | 213,411 (killed) | **691,417** |
| outcome | aborted after 4 tool calls | 7 full-file reads, exit 0, 13m34s, 80 requests |

Live Models API: `claude-fable-5 max_input_tokens=1,000,000`. Delivery
verified: a fresh `claude -p` with **no** CLI env var reports
`CLAUDE_CODE_AUTO_COMPACT_WINDOW=1000000`, i.e. the `settings.json` `env` block
propagates into the tool subprocess environment (it does NOT appear in
`/proc/<pid>/environ` — that is a false-negative check, do not use it).

## Proposed change

1. **`.claude/settings.json`** — add the `env` block (file currently has none):

       "env": { "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "1000000" }

   Verify against the model's real native window rather than hardcoding 1M
   blindly for a future smaller-window model; 1M is correct for the current
   fable-5/opus-5 fleet.

2. **`.claude/rules/gotchas.md`** — new entry with: the symptom, the
   `compact_boundary` `preTokens` diagnostic (far below native = enrolled;
   near-native = normal), the override + its documented limitations
   (`DISABLE_AUTO_COMPACT=1` does NOT affect it; `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`
   is capped by `Math.min()`; reported to fail on some arm64 native builds — we
   are x64), the `/proc` false-negative, and **the fact that the harness error
   text is misleading** ("a file being read or a tool output is likely too
   large" was false in every measured case here; largest read 12k tokens).

3. **Correct the wrong-cause guidance.** CLAUDE.md's autocompact-thrash bullet
   and the `-lean` twin agent descriptions attribute this class to
   subagent fixed-overhead (spec + CLAUDE.md import tree + MCP schemas). That is
   NOT the cause here: the 977,125-peak fable-5 agent had a **higher** floor
   (233,432) than the victims, and #1336's own marker records a **lean twin
   thrashing too**. Add the reduced-window cause as the first thing to check,
   with the `compact_boundary` diagnostic to discriminate. Do not delete the
   fixed-overhead guidance — it remains a real (separate) class.

## Scope / surfaces

- Primary target: `.claude/settings.json`
- Also: `.claude/rules/gotchas.md`, `CLAUDE.md` (autocompact-thrash bullet),
  `.claude/rules/LESSONS.md` if a new rule file is added instead of a gotchas
  entry, and the `-lean` agent descriptions under `.claude/agents/`.
- OUT of scope: `~/.claude/settings.json` (operator-global, already applied
  by hand this session; the repo copy is what this task makes durable).

## Constraints / invariants

- Workflow-surface only. No experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` (no-flags) passes; ruff clean on touched files.
- `.claude/settings.json` must stay valid JSON — validate with a parse after
  editing (a malformed global settings file breaks every future session).
- Adding an `env` block must not disturb existing keys (`permissions`,
  `hooks`, `enabledPlugins`, `model`).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:`
  Provenance line — it MUST NOT auto-route its own subagents' workflow-fix
  candidates (recursion guard). Subagents still EMIT candidate blocks normally;
  the orchestrator parks them (`.claude/rules/workflow-fix-on-bug.md`
  § Recursion guard, and its brief-composer clause).

## Upstream references (all read this session)

- anthropics/claude-code#46331 — `tengu_amber_redwood` silently reduces the
  autocompact window; **closed "not planned"**; documents the resolution chain,
  the override, and its limitations.
- anthropics/claude-code#43989 — v2.1.92 regression, threshold reduced to 400k
  on Opus 4.6 (1M context). OPEN.
- anthropics/claude-code#60485 — thrashing on fresh sessions with ~41 KB of
  context; closed as duplicate. Confirms the error text misleads.
- anthropics/claude-code#54056 — auto-compact at ~367K on Opus 4.7 [1m].
- anthropics/claude-code#42394 — fires despite `DISABLE_AUTO_COMPACT=1`.
- anthropics/claude-agent-sdk-python#958 — same thrash in the SDK.

Caveat to carry: the documented arm reduces to ~60% of native (1M -> 600k), but
our observed threshold implies ~235-300k — a more aggressive arm than the issue
describes. The override sits above the experiment in the chain either way, which
is why the fix works regardless.

## Provenance

- workflow_fix_target: .claude/settings.json
- fingerprint: 77e9d6470b27

Originating user instruction: "run the infra task now" (2026-08-04), following a
live diagnosis + verified fix in this session.
