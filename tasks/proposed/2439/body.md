---
title: 'gotchas.md: early-exit pipe probes false-fail under set -o pipefail (find|head|grep
  -q SIGPIPE class, #2388 Pod B R4/R5)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-21T02:52:29Z'
has_clean_result: false
parent_id: 2388
origin_prompt: 'Auto-filed by the #2388 orchestrator under the workflow-fix-on-bug
  protocol: two crash-fix rounds (Pod B R4/R5) traced to a pipefail+SIGPIPE false-empty
  probe; gotchas.md lacks the entry.'
workflow: v1
---
# Add gotchas.md entry: early-exit pipe probes false-fail under `set -o pipefail`

## Goal

Add a `.claude/rules/gotchas.md` entry (shell/launcher-script section) documenting the pipefail+SIGPIPE false-failure class, so pod launcher authors and reviewers stop shipping the poisoned probe shape.

## The bug class (two crash rounds on #2388, 2026-08-21)

Under `set -o pipefail` (every pod launcher's `set -euo pipefail` preamble), any pipeline whose DOWNSTREAM stage exits early poisons the pipeline status even when the probe logically succeeds:

```bash
find <dir> -maxdepth 1 -type f | head -1 | grep -q . || { echo "empty"; exit 1; }
```

`grep -q` exits 0 at the first match and closes the pipe; if `find` is still mid-walk (guaranteed on a network FS with stat-per-entry costs, e.g. runpodfs), `find` takes SIGPIPE, exits 141, and pipefail makes 141 the pipeline status. The `||` branch fires on a POPULATED directory. The failure is timing-dependent: it passes on fast local FS (find finishes before head exits, output fits the 64 KB pipe buffer) and fails deterministically on slow network mounts — which is why it survives local review and dies in production.

On #2388 this produced a false "hallucination_extraction extracted empty" hard exit (Pod B R4, commit 53b4d50739 shipped a WRONG lag diagnosis first), reproduced across 11 retry probes while a concurrent pipefail-free SSH session listed all 345 files. Fixed in commit 318d3d9bafbecc878ad044cb9bf307d3f2f5ad55.

## The safe shapes

- Emptiness/existence probe: `[ -n "$(find <dir> -maxdepth 1 -type f -print -quit)" ]` — no pipe; find stops itself at the first hit.
- General rule: never put an early-exiting consumer (`head`, `grep -q`, `read`) downstream of a long-running producer inside a pipefail script's test position; either let the producer terminate itself (`-print -quit`, `grep -m1 -l`, `--max-count`) or capture to a variable/file first.
- Interactive-shell verification is misleading: manual probes run WITHOUT pipefail, so "works when I ssh in" does not clear the script's probe.

## Acceptance

- gotchas.md gains the entry (with the #2388 incident citation and both safe shapes).
- LESSONS.md index row updated if the gotchas.md trigger line changes (lint `--check-lessons-index` green).
- Optional (cheap, judgement call): a `workflow_lint.py` WARN-only grep for `| head -1 | grep -q` / `| grep -q .` inside `scripts/*.sh` files that set pipefail.
