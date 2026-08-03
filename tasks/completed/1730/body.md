---
title: 'daily-fix: gcloud missing from non-login PATH silently repor'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3cec01f4afca
- daily-auto-filed
created_at: '2026-07-27T07:20:14Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): gcloud resolves only on
  the login-shell PATH, so a non-login tool shell gets command-not-found and a live-compute
  inventory reports no GCE instances — a silent zero on a spend surface'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 1 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Document that `gcloud` resolves only through `/snap/bin` (a login-shell-only PATH entry)
and prepend it wherever a live-compute inventory calls `gcloud`, so a command-not-found
can never read as an empty GCE inventory.

## Workflow gap

- **Bug observed:** a live-compute inventory pass ran
  `gcloud compute instances list --configuration=eps-gcp` from an ordinary (non-login) tool
  shell and got a command-not-found, which the pass rendered as "no GCE instances"; the
  binary is at `/snap/bin/gcloud`, and `/snap/bin` is absent from the non-login PATH.
- **Why it is a workflow gap:** no rule or recipe on the workflow surface records the
  `/snap/bin` login-shell-only fact, so every agent shell that scans for live GCE spend can
  silently return a zero on a money-spending surface — exactly the surface the scan exists
  to guard.
- **Confidence (emitter):** high
- verified-at-filing: environment probed at compose time —
  `command -v gcloud` in the tool shell → **not found**; `ls -l /snap/bin/gcloud` →
  **`/snap/bin/gcloud -> google-cloud-cli.gcloud`** (present); `bash -lc 'command -v gcloud'`
  → **`/snap/bin/gcloud`**; `echo "$PATH"` → **no `/snap/bin` entry**.
  Absence-of-guard greps, per target: `grep -c 'snap/bin' .claude/rules/gotchas.md` → **0**;
  `grep -niE 'login[- ]shell' .claude/rules/gotchas.md` → **0**;
  `grep -n 'gcloud' .claude/skills/daily/SKILL.md` → **0**;
  `grep -rln 'snap/bin' .claude/ scripts/` → hits only under `.claude/worktrees/**`
  (unrelated `eval_results` JSON + a task `events.jsonl`), **0 in any live workflow-surface
  file**. Landed-fix check: `git log --oneline --since='7 days ago' -- .claude/rules/gotchas.md`
  → 6 commits, none touching PATH resolution. (2026-07-26)

**Context binding — one target corrected.** The mined report proposes prepending
`/snap/bin` "in the compute-inventory recipe in `.claude/skills/daily/SKILL.md`". There is
no gcloud recipe in that file (`grep -n 'gcloud'` → 0 hits). The live-compute scan recipe
naming `gcloud compute instances list --configuration=eps-gcp` lives in `CLAUDE.md` (the
"Verify before asserting" bullet, L187). `.claude/skills/daily/SKILL.md` remains a valid
secondary surface because its § Inputs "Useful commands" block already carries the exact
sibling line `export PATH="$HOME/.local/bin:$PATH"   # uv lives in ~/.local/bin; non-login
(cron) shells miss it` (L35, repeated at L609) — the same class of fact, one entry short.

Related, already landed, and NOT the gap: `.claude/rules/background-automation.md` L117-124
documents that the GCP janitor CLI runs its own list-preflight and exits **3**
(`list-failed`) on a non-zero `gcloud ... list` rc, precisely so a disarmed janitor cannot
read green. That guard covers the cron path only; the interactive / agent-shell path this
filing addresses has no equivalent.

## Evidence

- Session `c0a2df1b`, 2026-07-26T06:33:14Z → 06:33:34Z: the live-compute inventory produced
  `"=== LIVE GCE INSTANCES ===\ntimeout: failed to run command 'gcloud': No such file or
  directory"`, then `"gcloud not found on PATH: ls: cannot access
  '/home/thomasjiralerspong/google-cloud-sdk': No such file or directory"`, then
  `"/snap/bin/gcloud"`. Once the binary was located the inventory returned 3 real
  `eps-issue-*` instances (all TERMINATED).
- Measured cost: 3 extra tool calls, roughly 20 s, in the session that noticed. The
  latent cost is the failure mode that does not notice: any fleet-burn or GCE-spend pass
  that treats a command-not-found as an empty list reports "no GCE instances" from a failed
  command.

## Proposed change

- `.claude/rules/gotchas.md` — add an entry in the cluster-specific-paths family (the file
  already carries `Hard-coded library paths in orchestrate/env.py — cluster-specific.` at
  L37): `gcloud` lives at `/snap/bin/gcloud`; `/snap/bin` is on the LOGIN-shell PATH only,
  so a non-login tool/cron shell gets `No such file or directory`. Use
  `export PATH="/snap/bin:$PATH"` (or `bash -lc`) before any `gcloud` call, and treat a
  non-zero `gcloud ... list` rc as UNKNOWN, never as an empty inventory.
- `.claude/skills/daily/SKILL.md` § Inputs "Useful commands" (L35; the same export recurs at L609) — extend the existing
  non-login PATH export line to cover `/snap/bin` alongside `$HOME/.local/bin`, with the
  same one-clause reason.
- `CLAUDE.md` "Verify before asserting" live-compute scan bullet (L187) — where it names
  `gcloud compute instances list --configuration=eps-gcp`, state that the call runs with
  `/snap/bin` on PATH and that a non-zero rc is not a zero result.
- Do not add a `gcloud` wrapper script or alias: the fix is PATH + rc discipline, and a
  wrapper would need its own maintenance while the janitor's landed list-preflight already
  covers the cron path.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- `.claude/skills/daily/SKILL.md` (§ Inputs, Useful commands PATH export)
- `CLAUDE.md` (live-compute scan bullet, L187)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 3cec01f4afca

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: C-P4.
