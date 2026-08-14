---
title: 'daily-fix: preflight disk probe uses writable dir'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-08-03T07:03:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): The preflight disk-quota
  probe attempts to posix_fallocate a temp file at the filesystem ROOT (''/.preflight_disk_probe.<pid>...'')
  when checking ''/'' -- Permission denied for a non-root user on the VM, so the probe
  silently degrades to shutil.disk_usage on EVERY VM run (''Could not run disk-quota
  probe on /: [Errno 13] Permission denied'', session f9af5f4c 21:35:31Z; code probed
  at compose time: prefl'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miner6, session f9af5f4c, task #1336).

## Goal

The EDQUOT/fallocate probe actually runs on the VM instead of silently degrading every preflight.

## Workflow gap

- **Bug observed:** The preflight disk-quota probe attempts to posix_fallocate a temp file at the filesystem ROOT ('/.preflight_disk_probe.<pid>...') when checking '/' -- Permission denied for a non-root user on the VM, so the probe silently degrades to shutil.disk_usage on EVERY VM run ('Could not run disk-quota probe on /: [Errno 13] Permission denied', session f9af5f4c 21:35:31Z; code probed at compose time: preflight.py:568 builds the probe path directly under check_path).
- **Why it is a workflow gap:** n/a -- experiment/infra code fix (orchestrate/ is outside the workflow surface); filed for independent review as a non-wf-fix daily item.
- **Confidence (emitter):** high (warn line probed by miner; code path read at compose time: Path(check_path)/'.preflight_disk_probe...' at preflight.py:568)
- verified-at-filing: `grep -n 'preflight_disk_probe' src/explore_persona_space/orchestrate/preflight.py` -> 1 hit (line 568, path built under check_path with no writable-dir fallback).

## Proposed change (refine in planning)

probe a user-writable directory on the SAME filesystem (repo root or $HOME when they resolve to the checked mount) instead of the mount root when the process is not root.

## Scope / surfaces

- Primary target: `src/explore_persona_space/orchestrate/preflight.py`

## Constraints / invariants

- Workflow-surface rules apply (experiment-code fix; NOT the workflow surface -- wf_fix: false); `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.


