---
title: preflight behind-main check is branch-unaware — hard-gating launchers can never
  pass on issue-branch pods
kind: infra
tags: []
created_at: '2026-06-10T22:22:32Z'
has_clean_result: false
---
## Problem

`src/explore_persona_space/orchestrate/preflight.py:141-149`'s behind-origin/main check reports `ok=false` for any pod checked out on an `issue-<N>` branch, with no branch awareness. Launchers that hard-gate on preflight's exit code under `set -euo pipefail` (e.g. `scripts/run_issue552_emresp_followup.sh` Step P) can never pass on an issue-branch pod.

## Incident

#552 follow-up launch (2026-06-10): the experimenter had to convert the pod to a single-branch clone (narrowed fetch refspec + dropped the origin/main tracking ref) to make preflight pass — a workaround, not a fix.

## Fix sketch

When `HEAD` is on a branch matching `issue-\d+` (or any non-main branch with an upstream of the same name), compare against `origin/<branch>` instead of `origin/main`, or downgrade behind-main to a WARNING on non-main branches.

## Acceptance criteria

1. preflight returns ok=true on a pod checked out at the tip of a pushed issue-branch.
2. Still fails when the issue branch itself is behind its own origin ref.
3. Unit test covering both.
