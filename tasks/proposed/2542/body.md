---
title: 'audit_clean_results_body_discipline: documented bare-invocation gate command
  is a broken legacy bulk mode'
kind: infra
tags: []
created_at: '2026-08-24T14:46:47Z'
has_clean_result: false
parent_id: 823
origin_prompt: 'Surfaced by the /issue 823 orchestrator at clean-result round 10:
  the documented bare gate invocation dies with FileNotFoundError on a missing inventory
  cache; the working form is --issue <N>.'
workflow: v1
---
## Goal

Fix the documented clean-result gate command for
`scripts/audit_clean_results_body_discipline.py`: the bare no-argument form named
in `CLAUDE.md` and in the agent specs is a legacy BULK mode that cannot run.

## Why

Found at #823 round 10. The bare invocation reads
`.claude/cache/audit-2026-05-08/inventory.json`, a pre-built inventory from a bash
paginator. That file does not exist on this VM (the directory does; the file does
not), so the bare form dies:

```
FileNotFoundError: [Errno 2] No such file or directory:
  '.claude/cache/audit-2026-05-08/inventory.json'
```

The working single-body form is `--issue <N>` (an argparse alias of `--task`),
which exits 0 with "PASS: no body-discipline anti-patterns matched".

The consequence is worse than a broken command. `CLAUDE.md`'s gate summary, the
`analyzer` and `clean-result-critic` specs, and every brief composed from them name
the bare form. So any agent that followed the documented command either silently
substituted the right flag or reported a PASS it never obtained. A gate whose
documented invocation cannot run is a gate that is sometimes not run at all, and
the failure is silent from the orchestrator's side because the agent reports PASS.

## Scope

Pick one and make the documentation match:

- Make the bare form default to a useful mode (e.g. error out with a usage hint
  naming `--issue <N>`, rather than a raw `FileNotFoundError` from a missing
  cache), AND
- Update every surface that names the bare form to name `--issue <N>`:
  `CLAUDE.md`'s gate summary, `.claude/agents/analyzer.md`,
  `.claude/agents/clean-result-critic.md`, `.claude/agents/codex-clean-result-critic.md`,
  and any `.claude/skills/issue/steps/*` occurrence.

Consider whether the legacy bulk mode should be retired outright, or gated behind
an explicit `--bulk` flag so it can never be reached by omission.

## Acceptance

- The bare invocation either works or fails with an actionable usage message
  naming the correct form; it never raises a bare `FileNotFoundError`.
- `grep -rn 'audit_clean_results_body_discipline.py' CLAUDE.md .claude/` shows no
  remaining bare-form gate instruction.
