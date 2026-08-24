---
title: 'workflow-fix: codex-composer-common mem-commit message file collides across
  parallel same-name composers'
kind: infra
tags:
- wf-fix
created_at: '2026-08-24T21:33:33Z'
has_clean_result: false
origin_prompt: workflow-fix-candidate emitted by the codex-critic alternatives-lens
  composer during /issue 2552 Phase 2 (2026-08-24); corroborated by the statistics-lens
  composer's overwritten commit message (cd03b05736).
workflow: v1
---
# workflow-fix: codex-composer-common.md mem-commit message file collides across parallel same-name composers

## Goal

Make the agent-memory commit recipe in `.claude/rules/codex-composer-common.md` § "Your own agent-memory writes" collision-safe for parallel composer batches.

## Symptom (observed 2026-08-24, #2552 Phase 2 round 1)

The recipe names the commit-message file `/tmp/mem-commit-<your-agent>.txt`, keyed by AGENT NAME only. The three `codex-critic` lens composers run in PARALLEL in one Phase-2 spawn batch and share the agent name, so they collide on the same message-file path. Observed twice in one batch:

- The Statistics-lens composer's `git commit -F /tmp/mem-commit-codex-critic.txt` landed with its message OVERWRITTEN by the Alternatives-lens composer's just-written content (commit `cd03b05736` — content correct, message belongs to the sibling; the Statistics composer itself reported the mismatch).
- The Alternatives-lens composer independently reported the same race from its side (its commit `cf487a3ca7`).

Whichever composer writes last silently supplies the OTHER's commit message (message/content mismatch in the durable history), and a Write racing a mid-read `-F` is possible. The collision is SILENT — no error, unlike the index.lock case the recipe already covers.

## Provenance

workflow_fix_target: .claude/rules/codex-composer-common.md
Surfaced by: codex-critic composers (alternatives + statistics lenses) during /issue 2552 Phase 2, session 2026-08-24. Fingerprint: mem-commit-message-file-collision-parallel-composers.

## Fix sketch

Change the recipe's named path to a per-invocation unique form, e.g. `/tmp/mem-commit-<your-agent>-<lens-or-role-suffix>-$$.txt` (or any distinguishing suffix the brief provides), and add one line noting that same-named composers run in parallel batches so the message file must be invocation-unique. The index.lock retry guidance already covers the sibling-holder case; the message-file collision is silent, so the path fix is the load-bearing part.

## Acceptance criteria

1. The recipe's message-file path in `.claude/rules/codex-composer-common.md` is invocation-unique (a `$$` pid suffix or equivalent).
2. One sentence in the recipe names the parallel-batch collision class as the reason.
3. `workflow_lint.py` no-flags run stays green; any test pinning the recipe text updated.
