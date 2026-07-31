---
title: 'daily-fix: pre-commit hook for agent-memory index size'
kind: infra
tags:
- wf-fix
- wf-fix-fp:66ff9973db84
- daily-auto-filed
created_at: '2026-07-31T06:58:55Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): the new --check-agent-memory-index-size
  lint (landed via #1891) only fires at worktree-mediated gates, so direct-to-main
  memory-save commits (the primary regrowth channel) are ungated, unlike the agent-spec
  sibling hook from #1661.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 Step C (parked workflow-fix-candidate routing) from a prose follow-up parked on task #1891 (emitting source: #1891's session, recursion-guarded; parked 2026-07-31T00:31:24Z). #1891 itself landed the `--check-agent-memory-index-size` lint check + curation (merged 28bb5842dcac); this is the pre-commit parity leg it deferred.

## Goal

Add a `.pre-commit-config.yaml` local hook running `workflow_lint.py --check-agent-memory-index-size` so direct-to-main memory-save commits are gated at commit time, matching the agent-spec sibling hook (#1661).

## Workflow gap

- **Bug observed:** MEMORY.md regrowth is mostly direct-to-main memory-save commits that no worktree gate lints — the new `--check-agent-memory-index-size` check (landed via #1891) only fires at worktree-mediated gates, so a direct root-side memory append can push an index back over the 24,000-byte FAIL threshold with no local signal until a later unrelated gate goes red.
- **Why it is a workflow gap:** the check's agent-spec sibling gained a local pre-commit hook for exactly this direct-to-main offender class (#1661); the new agent-memory check lacks that parity, leaving its primary regrowth channel ungated.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'check-agent-memory-index-size\|check_agent_memory_index_size' scripts/workflow_lint.py` → 5 hits (the check is landed on main); `grep -n 'agent-memory' .pre-commit-config.yaml` → 0 hits (the hook is absent) (2026-07-31 filing time). #1891 merged 2026-07-31T03:51:57Z as 28bb5842dcac.

## Proposed change (candidate diff sketch — refine in planning)

Add a .pre-commit-config.yaml local hook running `workflow_lint.py --check-agent-memory-index-size` (files regex `.claude/agent-memory/.*/MEMORY\.md` + `scripts/workflow_lint.py`, pass_filenames false), plus a hook-coverage pin test.

## Scope / surfaces

- Primary target: `.pre-commit-config.yaml`
- Model on the #1661 agent-spec sibling hook; keep hook runtime bounded (the check is file-size based, fast).

## Constraints / invariants

- Workflow-surface only. The hook must not slow every commit materially (scope via files regex).
- `scripts/workflow_lint.py --check-references` passes after the change; add the hook-coverage pin test.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .pre-commit-config.yaml
- fingerprint: 66ff9973db84 (tag-authoritative; supersedes body-carried fingerprint: 6fc2af2d68e4)
- origin: parked candidate-block on #1891 events.jsonl, ts 2026-07-31T00:31:24Z (routed by /daily 2026-07-30 Step C)
