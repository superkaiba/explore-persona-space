---
title: 'daily-fix: document HTML-escaped bg-notification recipe'
kind: infra
tags:
- wf-fix
- wf-fix-fp:635ca532d26f
- daily-auto-filed
created_at: '2026-07-14T06:44:53Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-13 problem sweep (route 2): background-agent completion
  notification bodies arrive HTML-escaped and the recipe (read the output file instead)
  is undocumented - sessions #1287 and #1288 each independently rediscovered the workaround'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-13 from the transcript problem sweep (sessions c338f95f → #1287 and 5cbde72b → #1288, same friction ~07:09Z and ~07:33Z).

## Goal

Document once, in the `/issue` skill's subagent-dispatch/notification handling, the working recipe for HTML-escaped background-agent completion notification bodies: read the agent's report/plan text from its output file (or the durable artifact it wrote) instead of the notification body.

## Workflow gap

- **Bug observed:** background-agent completion notification bodies arrive HTML-escaped in the harness; on 2026-07-13 sessions #1287 and #1288 each independently rediscovered the workaround (extracting clean plan text from the agent output file — the "transcript recipe") because it is documented nowhere in the workflow surface.
- **Why it is a workflow gap:** the harness behavior is not project-fixable, but the recovery convention is — leaving it undocumented costs each session a rediscovery and risks a session pasting escaped text into a durable artifact.
- **Confidence (emitter):** medium (recurring, cheap fix, doc-only)
- verified-at-filing: `grep -n "HTML-escap" .claude/skills/issue/SKILL.md` → 0 hits (the :4287/:4321/:4917 "escaped" hits are pod/trap prose, unrelated) (2026-07-14 UTC).

## Proposed change (candidate diff sketch — refine in planning)

One short paragraph in `.claude/skills/issue/SKILL.md` near the subagent-dispatch/notification-handling prose:

```
+ Background-agent completion notification bodies may arrive HTML-escaped.
+ Never paste notification-body text into plans/markers/artifacts — read the
+ agent's report from its durable output (the file the brief told it to
+ write, or its output file) instead.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`

## Constraints / invariants

- Doc-only; `scripts/workflow_lint.py` default run passes.
- Recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 635ca532d26f

Origin: transcript-mined (c338f95f ~07:33Z, 5cbde72b ~07:09Z). Not a parked candidate — surfaced by the /daily problem sweep.
