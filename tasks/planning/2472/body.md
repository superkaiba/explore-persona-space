---
title: 'Add codex-*-lean twins: autocompact-thrash ladder has no lean-twin rung for
  the Codex composer roles'
kind: infra
tags:
- wf-fix
created_at: '2026-08-22T14:35:21Z'
has_clean_result: false
workflow: v1
---
---
kind: infra
workflow_fix_target: .claude/rules/context-hygiene.md
fingerprint: codex-composer-roles-have-no-lean-twin-autocompact-ladder-terminates
---

# Autocompact-thrash ladder has no lean twin for the `codex-*` composer roles

## Goal

Close the escalation gap in `.claude/rules/context-hygiene.md`'s
autocompact-thrash ladder (Class 2, fixed-overhead subagent-window pressure) for
the `codex-*` prompt-composer roles, which currently have no lean twin and
therefore fall off the ladder one rung early.

## The gap

`context-hygiene.md`'s Class-2 recipe says: respawn micro-scoped on the default
model; if that ALSO thrashes, "escalate ONCE (same `v<n>`, no counter increment)
to the role's LEAN TWIN with the same brief", and it enumerates six:
`analyzer-lean`, `planner-lean`, `critic-lean`, `experiment-implementer-lean`,
`code-reviewer-lean`, `consistency-checker-lean`.

There is no lean twin for ANY `codex-*` role — not
`codex-code-reviewer`, `codex-critic`, `codex-interpretation-critic`,
`codex-clean-result-critic`, or `codex-follow-up-critic`. So when a Codex
composer thrashes, the ladder runs out after the micro-scoped default-model
respawn, and the only remaining moves are the fail-loud terminal or refusal
rung (c)'s inline composition by the orchestrator.

## Why it matters

Inline composition is sanctioned but expensive in a specific way: it breaks the
review site's independence guarantee, which is why it REQUIRES the
`[epm-inline-fallback] role=<role> round=<n> reason=<one-line>` progress marker
(#2062). Every Codex composer thrash that reaches this point converts a doubled
review site into a single-Claude one. The lean-twin rung exists precisely to
avoid paying that price, and the composer roles — the ones most likely to thrash,
since a composer inlines a full lens reference or SPEC into its prompt — cannot
reach it.

The composers are also the CHEAPEST roles to lean-twin: they are thin
prompt-composers whose whole job is Read + Write + return a path. They need no
MCP tools at all, and an all-tools spawn loads ~138K tokens of MCP schemas
(measured 2026-07-03) that a composer never uses.

## Observed

Two Codex composers in one `/issue 2291` Step 5 round died degenerate at
~135k tokens with ZERO tool calls — the Class-2 signature (no oversized tool
result to bound; fixed overhead alone exhausted the window). The round recovered
because the Claude reviewer's verdict stood and a later composer attempt
succeeded, so no inline fallback was needed — but the ladder had no rung left to
offer.

## Scope

- Add `codex-*-lean` twins for the five composer roles, following the existing
  lean-twin pattern: restricted tool list (`Bash, Read, Write, Edit, Grep, Glob`
  — no `mcp__*`, no `skills:`), deferring to the full composer spec by reference
  rather than restating it.
- Resolve them user-global in `~/.claude/agents/` with repo symlinks, matching
  the #2072 placement rule (agent types register at session start from the
  session cwd + user-global dirs, never mid-session).
- Extend `context-hygiene.md`'s lean-twin roster to name them, so the ladder
  text and the available agent set agree.
- Cross-check `.claude/rules/codex-ensemble-review.md`: its composer-death
  handling should point at the new rung before the no-show fallback.

## Non-goals

- Do NOT lean-twin the Codex DISPATCH path (`scripts/codex_task.py`) — that is
  the orchestrator's bg-Bash call, not a subagent, and has no window of its own.
- Do NOT pin a smaller model as the fix. That is the INVERSE of refusal rung
  (b2) and is explicitly banned for thrash (#1090: default-model spawns
  compacted fine while 3/6 sonnet spawns thrashed).

Provenance: surfaced during `/issue 2291` Step 5 (two degenerate Codex composer
spawns); held during the round to avoid a snapshot-drift race with the live
reviewers, filed at Step 10 close. Prose follow-up per
`.claude/rules/workflow-fix-on-bug.md`.
