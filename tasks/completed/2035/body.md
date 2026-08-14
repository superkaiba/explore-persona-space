---
title: 'daily-fix: backend_poll reachable-wedge escalation arm'
kind: infra
tags:
- wf-fix
- wf-fix-fp:554f10058988
- daily-auto-filed
created_at: '2026-08-03T07:01:04Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): GCE leg newarma5syc (#1739)
  sat futex-wedged ~21.6h (6.5 CPU-minutes total, GPU 0%, workload log stale, phase
  frozen at a non-terminal value) while ~91 poll ticks read it healthy ''running'';
  caught only by a manual probe from the interactive session. The existing #669 frozen-phase
  arm requires a REACHABILITY alarm conjunct, which a reachable-but-wedged box never
  trips; poll JSON showed gpu_util ''un'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miner1, sessions f98a12ed/20e82ec2, task #1739).

## Goal

Close the reachable-wedge blind spot in backend_poll's GCP wedge escalation: a frozen non-terminal `eps/phase` + stale workload-log mtime + budget overrun escalates to `terminal_workload_wedged` even when reachability is OK.

## Workflow gap

- **Bug observed:** GCE leg newarma5syc (#1739) sat futex-wedged ~21.6h (6.5 CPU-minutes total, GPU 0%, workload log stale, phase frozen at a non-terminal value) while ~91 poll ticks read it healthy 'running'; caught only by a manual probe from the interactive session. The existing #669 frozen-phase arm requires a REACHABILITY alarm conjunct, which a reachable-but-wedged box never trips; poll JSON showed gpu_util 'unknown' and no eta-deviation escalation past the 10h budget.
- **Why it is a workflow gap:** The #669 arm was built for guest-network death (unreachable); a futex/HF-download wedge keeps SSH/serial reachable so no existing conjunct fires, and the poller reads `running` forever -- the exact #667-class gap CLAUDE.md documents as pending.
- **Confidence (emitter):** high (incident probed by miner; gap read from backend_poll.py source at compose time)
- verified-at-filing: `grep -n -iE 'frozen|drain.timeout|reachab' scripts/backend_poll.py` -> #669 arm exists (escalate_frozen_phase, line ~1137) and its docstring REQUIRES the reachability alarm conjunct; no stale-log-mtime or budget-overrun conjunct exists (0 hits for either). Miner evidence probed: ~91 healthy tick tool_results 15:27Z->03:45Z; wedge marker epm:progress v347 on #1739 (2026-08-03T03:37:16Z) quotes 6.5 CPU-min/21.6h, wchan futex_wait_queue, GPU 0%.

## Proposed change (refine in planning)

extend the wedge escalation to fire on frozen non-terminal phase + stale workload-log mtime + budget overrun even when the box stays REACHABLE (and treat sustained gpu_util-unknown / CPU-seconds starvation as wedge evidence).

## Scope / surfaces

- Primary target: `scripts/backend_poll.py`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 554f10058988

- workflow_fix_target: scripts/backend_poll.py

