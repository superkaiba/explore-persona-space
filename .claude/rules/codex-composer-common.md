---
paths:
  - ".claude/agents/codex-*.md"
description: >
  The shared composer contract for every codex-* twin wrapper (the #533
  compose-only hard rule, companion location, temp-file write + validation
  bounds, and the return contract) — one canonical copy; each twin's spec
  keeps a 3-line pointer plus its role-specific deltas.
---

# Codex composer common contract (all codex-* twins)

Every `codex-*` agent is a THIN Claude prompt-composer for a Codex (gpt-5.5)
twin reviewer. One canonical copy of the shared contract lives here; the
per-twin specs carry only their role deltas (what to inline, the verdict
template, marker vs in-context output mode).

## Compose-only — NEVER dispatch Codex yourself

This is the load-bearing constraint for the entire wrapper-agent class.

- **You write a prompt to a temp file and return its path.** That is the
  whole job. The orchestrator (the conversation's parent loop) is the ONLY
  context that may dispatch Codex.
- **NEVER call** `scripts/codex_task.py` (with or without `--background` /
  `run_in_background=true`).
- **NEVER call**
  `node ~/.claude/plugins/cache/openai-codex/codex/*/scripts/codex-companion.mjs`
  with `companion task`, `--background`, or any spawn subcommand — the
  `companion task --background` form is the exact anti-pattern that creates
  orphan jobs.
- **NEVER spawn a polling loop** (`while`/`until` sleep over
  `codex-companion status`).
- The only Bash you may run: reading agent specs / lens references, reading
  the inputs your brief named, locating the companion script (sanity check
  only — do NOT execute it), writing the prompt file, and local prompt-file
  validation that reads/writes temp files only. Local validation MUST NOT
  invoke `codex_task.py` / `codex-companion.mjs` in any form, MUST NOT spawn
  a polling loop, and MUST NOT post any marker.
- **Why this matters.** A subagent has ONE turn. If you spawn Codex in-turn,
  the broker registers the job to your session, you exit, and the job has no
  listener for completion — it stays "running" forever from any other
  context's view, then becomes unqueryable when the broker garbage-collects
  the session. The harness delivers a bg-completion notification only to the
  orchestrator's own `Bash(run_in_background=true)` invocation; there is no
  workaround from inside a subagent turn. (Incident task #533, 2026-06-10,
  job `task-mq7kn6dp-fpu8xo`: the wrapper dispatched in-turn and exited; the
  orchestrator burned 42 minutes watching a dead handle before the no-show
  fallback.)

## Locate the companion (sanity check only)

Glob `~/.claude/plugins/cache/openai-codex/codex/*/scripts/codex-companion.mjs`
(any version dir). Found ⇒ proceed to compose — never execute it. Missing
(plugin upgrade race, cache wipe) ⇒ **do NOT try to "make it work"**: print
`BLOCKER: codex companion missing` to stdout and exit; the orchestrator
falls back to the single-Claude decision for the affected site/lens.

## Temp-file write + validation

Write the composed prompt to the exact output path your brief/spec names
(`/tmp/codex-*.md` convention), substituting every `{{...}}` placeholder.
Validate LOCALLY before returning: no unsubstituted `{{...}}` residue
outside deliberately-kept placeholder lines your spec names, required
envelopes/sections present, and any role-specific checks (numeric-leak
verifier for plan-critic twins, envelope validation for the code-review
twin) — all read/write temp files only.

## Return contract

Return the prompt-file path (plus the fields your spec's return template
names) as your final text. You never dispatch, never poll, never post
markers — the orchestrator dispatches `scripts/codex_task.py` as bg Bash and
posts the verdict marker from the output file.
