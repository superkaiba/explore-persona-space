---
title: 'daily-fix: guard waiver for gcloud ssh --command payloads'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f69bc770b5bf
- daily-auto-filed
created_at: '2026-07-17T06:58:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the repo-root guard false-positive-BLOCKs
  git verbs inside REMOTE-execution payloads whose clause head is gcloud (gcloud compute
  ssh ... --command with a git merge --ff-only inside, #825 ~13:18Z) — the clause
  waiver keys on command word ssh/grep-family only and the #1413 mask covers ssh single-quoted
  payloads, not the gcloud form; #1336 (~03:51Z) hit the ssh variant pre-#1413; both
  sessions silent'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (recurrence 3 counting tonight's own heredoc block). a87f5898f6 (#1413) landed the ssh single-quote mask the same day — the gcloud clause-head form is the residual.

## Goal

Stop the repo-root guard from blocking git verbs that execute REMOTELY via gcloud compute ssh --command payloads (and weigh the prose/heredoc text-match residual).

## Workflow gap

- **Bug observed:** the repo-root guard false-positive-BLOCKs git verbs inside REMOTE-execution payloads whose clause head is gcloud (gcloud compute ssh ... --command with a git merge --ff-only inside, #825 ~13:18Z) — the clause waiver keys on command word ssh/grep-family only and the #1413 mask covers ssh single-quoted payloads, not the gcloud form; #1336 (~03:51Z) hit the ssh variant pre-#1413; both sessions silently split commands as a workaround, and tonight's /daily run itself was blocked composing a file whose PROSE contained git-verb strings (heredoc text-match)
- **Why it is a workflow gap:** The guard should key on git commands executing against the LOCAL repo root; remote payload text and quoted prose are out of its jurisdiction, and silent split-command workarounds hide the class.
- **Confidence (emitter):** medium-high
- verified-at-filing: `grep -c gcloud scripts/guard_repo_root_branch.sh` -> 0 (no gcloud clause handling — absence claim); ssh waiver + #1413 mask present (L49-L108, L247); landed-fix check: a87f5898f6 masks ssh payloads only; reproduced live tonight (this run's blocked heredoc, 2026-07-17 ~07:1xZ)

## Proposed change (candidate diff sketch — refine in planning)

extend the remote-execution clause waiver (or the #1413 mask pre-pass) to gcloud compute ssh --command payloads, with the same residual-closure discipline as the ssh arm; consider the heredoc/prose text-match residual while in the file

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: f69bc770b5bf

