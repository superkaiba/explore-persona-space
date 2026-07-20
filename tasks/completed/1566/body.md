---
title: 'daily-fix: guard FP on git literals in --note payloads'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2348cda7b54d
- daily-auto-filed
created_at: '2026-07-20T06:48:18Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): post-marker --note quoting
  ''git checkout -b'' blocked though nothing mutated'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-19 (route 2) from transcript-mined problems (see evidence in ## Provenance).

## Goal

Stop `scripts/guard_repo_root_branch.sh` from false-positive-blocking commands whose git-mutation literals appear only inside QUOTED ARGUMENT PAYLOADS (e.g. `task.py post-marker --note '...git checkout -b...'`), e.g. by masking quoted `--note`/`--title` argument bodies before pattern matching (the #1413 `mask_ssh_payload_separators` pre-split masking pass is the in-file precedent).

## Workflow gap

- **Bug observed:** the hook BLOCKED a `task.py post-marker` call because the marker NOTE text quoted the literal `git checkout -b` (clarifier context notes about the guard itself); the Bash command mutated nothing, and the session had to detour via a /tmp file + `post-marker --file`.
- **Why it is a workflow gap:** the hook's regex matches trigger literals anywhere in the command string, so any marker/note text that DISCUSSES git commands trips it — a growing false-positive class as guard-related workflow-fix tasks proliferate.
- **Confidence (emitter):** medium
- verified-at-filing: semantic probe: `tests/test_guard_repo_root_branch.py:36` documents the existing #1413 pre-split masking pass for SSH payloads (context read: masking exists for ssh remote strings, NOT for --note/--title quoted payloads — absence claim scoped to the payload class). Incident: session 90ac34d0 (task #1545) @ 15:50 UTC 2026-07-19, BLOCKED text quoting 'git checkout -b' inside a --note payload; self-corrected via post-marker --file.

## Proposed change (candidate diff sketch — refine in planning)

(none — sketch: extend the pre-split masking pass to mask quoted argument bodies following --note/--title/--origin-prompt flags of task.py invocations before the trigger-literal scan; keep fail-closed on anything outside a recognized quoted-payload shape)

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`
- Tests: `tests/test_guard_repo_root_branch.py` (the masking-pass test family).

## Constraints / invariants

- Workflow-surface rules apply where the target is workflow surface; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard applies where tagged wf-fix (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: scripts/guard_repo_root_branch.sh
- fingerprint: 3a517bd4efd2

Mined evidence: PreToolUse block: "BLOCKED: 'git checkout -b' would move the SHARED repo-root tree off main" on a post-marker call whose --note quoted guard deny-text (session 90ac34d0, #1545, 2026-07-19). NOTE: this session was under the recursion guard and did not even park a candidate — sweep-derived.
