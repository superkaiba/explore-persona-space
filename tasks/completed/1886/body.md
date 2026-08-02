---
title: 'workflow-fix: gotchas cross-machine-seam entry — SLURM rsync lane drops committed
  eval_results'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ee28c59127ed
created_at: '2026-07-30T15:58:59Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised by the #1689 crash-fix implementer (2026-07-30):
  the gotchas.md cross-machine-seam entry names only gitignored data/ on git-clone
  lanes; extend it to the SLURM rsync lane''s eval_results/ exclusion (doc-side sibling
  of open #1835)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1689 (emitting agent: experiment-implementer, crash-fix round 5).

## Goal

Extend the "Every file an OFF-POD phase loads must be in the pod's UPLOAD SET" gotchas entry to name the SLURM rsync lane's eval_results/ exclusion: committed-in-git eval_results inputs also need HF upload-first plus a leg-entry staging step there.

## Workflow gap

- **Bug observed:** The #1689 fellows job 15724 crashed FileNotFoundError because the SLURM lane's rsync excludes eval_results/ wholesale — COMMITTED parent eval_results inputs are not node-reachable, but the cross-machine-seam gotcha entry (#1482/#1773) only names gitignored data/ on git-clone lanes.
- **Why it is a workflow gap:** The documented seam rule lets an author conclude "committed to the branch ⇒ reaches the node", which is false on the rsync (fellows/SLURM) lane specifically for eval_results/ (and every other RSYNC_EXCLUDE dir). Doc-side sibling of open #1835 (which mechanizes the gate in verify_carryover_inputs.py + adds the extra-sync knob); this candidate covers the RULE text so plan/implement-time readers stop inheriting the false premise before #1835's mechanical gate lands.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'UPLOAD SET\|upload set' .claude/rules/gotchas.md` → the cross-machine-seam entry present with no SLURM/rsync/eval_results clause (checked 2026-07-30); `RSYNC_EXCLUDE_PATTERNS` in `src/explore_persona_space/backends/slurm.py` confirmed excluding `eval_results/`.

## Proposed change (candidate diff sketch — refine in planning)

```
+ The seam also covers COMMITTED eval_results inputs on the SLURM rsync lane:
+ RSYNC_INCLUDE_PATHS (backends/slurm.py) excludes eval_results/ wholesale, so
+ "committed on the branch" is NOT node-reachable there (#1689 job 15724:
+ FileNotFoundError in cmd_fence); remedy = #734 upload-first + an idempotent
+ fail-loud leg-entry staging step (worked example: issue1689_derived_vs_free.py
+ --phase stage-parent-inputs).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep before editing (`grep -rn 'UPLOAD SET' .claude/`) — one entry expected; keep the LESSONS.md index row's trigger phrasing in sync if the entry's fires-when clause changes.

## Constraints / invariants

- Workflow-surface only; `workflow_lint.py` no-flags run passes (gotchas.md size/format checks).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: ee28c59127ed

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: The #1689 fellows job 15724 crashed FileNotFoundError because the SLURM lane's rsync excludes eval_results/ wholesale — COMMITTED parent eval_results inputs are not node-reachable, but the cross-machine-seam gotcha entry (#1482/#1773) only names gitignored data/ on git-clone lanes.
why_workflow_gap: The documented seam rule lets an author conclude "committed to the branch ⇒ reaches the node", which is false on the rsync (fellows/SLURM) lane specifically for eval_results/.
proposed_change: Extend the "Every file an OFF-POD phase loads must be in the pod's UPLOAD SET" entry to name the SLURM rsync lane's eval_results/ exclusion: committed-in-git eval_results inputs also need HF upload-first + a leg-entry staging step there.
diff_sketch: |
  + The seam also covers COMMITTED eval_results inputs on the SLURM rsync lane:
  + RSYNC_INCLUDE_PATHS (backends/slurm.py) excludes eval_results/ wholesale, so
  + "committed on the branch" is NOT node-reachable there (#1689 job 15724:
  + FileNotFoundError in cmd_fence); remedy = #734 upload-first + an idempotent
  + fail-loud leg-entry staging step (worked example: issue1689_derived_vs_free.py
  + --phase stage-parent-inputs).
confidence: medium
related_task: #1689
<!-- /workflow-fix-candidate -->
