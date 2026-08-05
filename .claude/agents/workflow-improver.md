---
name: workflow-improver
description: >
  DEPRECATED (#678) — DO NOT SPAWN. Workflow-surface fixes are filed as
  `kind: infra` tasks and implemented by a background `/issue <N> --auto`
  session; see `.claude/rules/workflow-fix-on-bug.md`.
tools:
  - Read
model: "claude-fable-5"
---

Retired 2026-06-27 (#678); never spawned since. The workflow-fix-on-bug
protocol replaced this agent's auto-spawn with a filed `kind: infra` task plus
a background `/issue <N> --auto` session running the full code-change
pipeline. The `name:` is kept so any stale spawn of this type fails loud
rather than silently mis-routing.
