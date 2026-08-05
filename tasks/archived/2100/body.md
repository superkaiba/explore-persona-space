---
title: 'workflow-fix: verify_plan check — mount-binding + headroom for VM staging
  rows'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9d90991da605
created_at: '2026-08-05T21:03:55Z'
has_clean_result: false
origin_prompt: 'Mechanizable critic finding from task #2091 Phase 2: plans v2+v3 cited
  the non-live #681 worktree bind for a 42 GB staging row and verify_plan.py PASSed
  both — add a findmnt/df headroom check for section-9 disk rows citing the bind or
  /mnt/eps-data.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `mechanizable: yes` critic finding raised on task #2091 (emitting agent: methodology critic, Phase 2 plan review).

## Goal

Add a verify_plan.py check: a plan §9 disk row citing the worktree bind (or `/mnt/eps-data` staging) must be backed by a live mount/headroom probe — assert `findmnt --mountpoint <repo>/.claude/worktrees` non-empty when the row claims the bind, and flag when no `df -P`-style free-headroom figure ≥ ~1.5× the row's estimate is stated for the resolved filesystem.

## Workflow gap

- **Bug observed:** task #2091's plan v2 AND v3 cited the #681 worktree bind for a ~42 GB P4 staging row ("data/issue_2091/hf_dl/ resolves to /mnt/eps-data via the #681 worktree bind — never /"); the bind is not live on this VM (findmnt empty, verified 2026-08-05) and no candidate filesystem had 1.5× headroom (/ 21 GB free, /mnt/eps-data 47 GB free), yet verify_plan.py PASSed both versions (n_fail=0) — it has no mount-binding or headroom check, so the false placement premise reached the critic round and was caught only by a live probe there. Undetected, the run would have died at P4 entry AFTER the ~9 GPU-h pod spend + the ~60k-call judge wave.
- **Why it is a workflow gap:** the mechanical plan verifier is the designed catch-point for structurally-checkable false premises (the #1333 mount-binding class already has a named lens item in critic-lens-reference.md item 16, but nothing mechanical); a plan-time findmnt/df assertion is cheap, deterministic, and would have FAILed v2 at Phase 1.5.0 before any critic spawn.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'findmnt\|mountpoint\|eps-data' scripts/verify_plan.py` → 0 hits (absence-of-guard claim; empty output verified 2026-08-05); reproduced live: `findmnt --mountpoint <repo>/.claude/worktrees` → empty while plans/v3.md §9 cited the bind

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_c46_mount_binding_headroom(plan_text, ...):
+     # Fires when a §9/disk-row line cites the worktree bind or /mnt/eps-data
+     # as a staging/out-root target.
+     # FAIL when the row claims the #681 bind and
+     #   findmnt --mountpoint <repo>/.claude/worktrees is EMPTY on this VM.
+     # WARN when a /mnt/eps-data (or bind) staging row states no live free-space
+     #   figure (df -P) alongside its size estimate, or the stated free figure
+     #   is < 1.5x the row's estimated bytes.
+     # N/A escape: "N/A — no VM staging row" (no bind//mnt/eps-data citation).
```

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Also update: `tests/test_verify_plan.py` (fixtures for FAIL/WARN/N-A branches), the check-id list in `.claude/skills/adversarial-planner/SKILL.md` § canonical N/A escape phrases (add the new escape phrase), and `.claude/rules/plan-compute-sizing.md` if it names the check.
- Grep the workflow surface for prior mount-check discussion before editing (`grep -rn 'findmnt' .claude/ scripts/verify_plan.py`).

## Constraints / invariants

- Workflow-surface only. The check must be robust on pods/GCE (no `/mnt/eps-data` there — the check fires only on plans whose rows cite the VM bind/data-disk, and a non-VM invocation N/A-skips).
- `tests/test_verify_plan.py` passes; `scripts/workflow_lint.py` no-flags run passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 9d90991da605

Verbatim surfaced finding (methodology critic, task #2091 Phase 2 Must-Fix 1 tail): "mechanizable: yes — a plan-time check: for any §9 disk row citing the worktree bind, assert `findmnt --mountpoint <repo>/.claude/worktrees` non-empty AND `df -P` free ≥ 1.5× the row's estimate on the resolved filesystem."
