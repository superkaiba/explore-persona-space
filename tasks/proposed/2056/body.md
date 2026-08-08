---
title: 'workflow-fix: document PostToolUse formatter race (edit-then-run yields stale
  artifact, exits 0)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:posttooluse-formatter-race
created_at: '2026-08-04T00:11:03Z'
has_clean_result: false
origin_prompt: 'An agent edited a generator, ran it immediately, and got the OLD output
  with exit 0: the PostToolUse format hook rewrote the script ~2s after the run had
  already started against pre-edit bytes. Tell is stat showing the script mtime later
  than the artifact it produced. Not documented anywhere in .claude/rules/.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a concrete trap surfaced during task #1482 (emitting agent: `sae-map-coefficients`, 2026-08-04). It fails GREEN — exit code 0, stale artifact — which is the property that makes it worth a durable entry rather than a lesson one agent learned.

## Goal

Document the PostToolUse formatter / generator-run race in `.claude/rules/gotchas.md`, with its detection tell, so the next agent doing an edit-then-run cycle recognises a silent no-op instead of trusting the exit code.

## Workflow gap

- **Bug observed:** an agent edited a generator script, immediately ran it, and the run produced the OLD output while reporting success. The PostToolUse format hook rewrote the script roughly two seconds AFTER the generator had already executed against the pre-edit bytes. The agent caught it only by grepping the generated artifact for the specific new strings it expected; the exit code was 0 and nothing else was anomalous.
- **The detection tell:** `stat` shows the SCRIPT's mtime LATER than the artifact it supposedly produced. That ordering is impossible for a healthy run and is a one-command check.
- **Why it is a workflow gap:** the PostToolUse ruff/format hook is a workflow-surface behaviour (`.claude/settings.json`) that this repo applies to every Write/Edit, and the edit-then-immediately-run cycle is extremely common here — most per-issue analysis is "patch the driver, re-run it, read the figure". The hook's asynchrony relative to a following Bash call is not documented anywhere: `grep -rln "format hook|PostToolUse.*format|stale artifact.*mtime" .claude/rules/` returns nothing. The failure mode is silent and directional — it always yields the PREVIOUS behaviour, so an agent debugging a fix concludes the fix did not work and starts changing correct code.
- **Confidence (emitter):** medium-high. Observed once, with a clean mechanism and a cheap deterministic check; the race window is timing-dependent so reproduction may be intermittent.
- verified-at-filing: `grep -rln "format hook\|PostToolUse.*format\|stale artifact.*mtime" .claude/rules/` -> 0 hits (2026-08-04). `wc -c .claude/rules/gotchas.md` -> 311,958 bytes, so the addition must be terse. No similar open task: a `task.py list-by-status --status proposed` title scan for format-hook / stale-artifact / generator terms returned none.

## Proposed change (candidate diff sketch — refine in planning)

Add a short entry to `.claude/rules/gotchas.md`, roughly:

> **Edit-then-run can execute the PRE-edit script (PostToolUse formatter race).** The
> format hook rewrites a Write/Edit'd file asynchronously; a Bash run issued immediately
> after can execute the old bytes and exit 0 with a stale artifact. Signature: the fix
> appears to have no effect. Tell: `stat -c '%y %n' <script> <artifact>` shows the SCRIPT
> newer than the artifact it produced. Do not trust the exit code on an edit-then-run
> cycle — grep the artifact for a string only the new code emits.

Placement and exact wording are the planning session's call; keep it to a few lines given the file is already ~312 KB.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Consider whether the same tell belongs in `.claude/agents/experiment-implementer.md`, since implementers are the heaviest users of the edit-then-run cycle — but do NOT duplicate the prose; a pointer is enough.
- Do NOT change the hook itself. Making the formatter synchronous is a much larger change with its own risks, and is not what this task is asking for.

## Constraints / invariants

- Documentation only. No behavioural change to hooks, settings, or any script.
- `.claude/rules/gotchas.md` is already ~312 KB and on-demand-loaded; the addition must be terse enough not to worsen that. Check whether a size ratchet applies before expanding it.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: posttooluse-formatter-race-stale-artifact

Surfaced as a prose follow-up, verbatim:

> "A process note worth having. My first regeneration silently produced the *old* HTML: the format hook rewrote the script two seconds *after* the generator had already run against the pre-fix version, so the run reported success with stale output. I caught it only because I grepped the generated file for the specific new strings rather than trusting the exit code. The tell is `stat` showing the script's mtime later than the artifact it supposedly produced. Worth checking whenever an edit-then-run cycle appears to be a no-op." (sae-map-coefficients)
