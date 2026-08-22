---
title: 'set-body: fail-loud on byte-identical no-op + gate body-edit markers on an
  actual body commit (phantom v4 edits, #2333)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-18T08:16:48Z'
has_clean_result: false
origin_prompt: auto-filed by /issue 2333 orchestrator from the round-4 interpretation-critic's
  workflow-surface suggestion (durable marker asserted never-persisted body edits,
  2026-08-18)
workflow: v1
---
# Gate epm:interpretation-style body-edit markers on an actual body commit — a durable marker asserted never-persisted edits

## Goal

Close the phantom-body-edit channel surfaced in #2333 interpretation round 4: an analyzer posted `epm:interpretation v4` claiming two body edits applied + `verify_task_body PASS`, while `tasks/interpreting/2333/body.md` remained byte-identical to the prior commit (`929387386c`) — the edits existed in no durable state. Mechanism (forensically established): the body file's mtime moved at marker time but content was unchanged — `set-body` was handed a file that did not contain the edits (the agent edited one copy and applied another), git no-op'd on identical bytes, and the marker recorded success anyway. Same family as the #2034 unconfirmed-stand-down record: a durable claim written without probing the durable state it asserts.

## Fix (prescribed by the round-4 interpretation-critic)

At the verdict/marker-collection layer (candidate surfaces: `scripts/task.py post-marker` pre-post hook for `epm:interpretation` kinds, or the analyzer spec's landing-verification duty, or `workflow_lint`-class check in the /issue orchestrator's marker-collection step): an `epm:interpretation vN` (or any marker kind that CLAIMS body edits) must be preceded by a new set-body commit touching the task's `body.md` — marker-time > last-body-commit-time with a non-empty body diff — OR carry an explicit no-op declaration ("no body change this round"). Cheapest robust form: `set-body` itself should FAIL LOUD (or print a WARNING the caller must acknowledge) when the incoming file is byte-identical to the current body — the write is then a provable no-op and the agent learns immediately instead of at the next critic round. Also worth adding to `.claude/agents/analyzer.md`: the landing-verification duty (re-read the live body + confirm the new commit sha) after every set-body.

## Incident

2026-08-18, task #2333 interpretation micro-round 4 (marker 08:09:26Z; caught by the round-4 Claude critic 08:14:50Z via hash-object equality against the v3 commit; corroborated by the Codex composer's mtime forensics). Cost: one wasted ensemble round + a correction round.

## Candidate metadata

- target_file: scripts/task.py (set-body byte-identical no-op warning) + .claude/agents/analyzer.md (landing-verification duty)
- fingerprint: set-body-byte-identical-noop-plus-marker-body-commit-gate
- confidence: high (forensically reproduced; exact mechanism established)
