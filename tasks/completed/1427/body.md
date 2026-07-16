---
title: 'workflow-fix: gotchas.md — EXDEV: bare /tmp tempdir + os.replace onto /workspace'
kind: infra
tags:
- wf-fix
- wf-fix-fp:23d8f7eb7e71
created_at: '2026-07-16T14:39:42Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #1335 r9 crash-fix (EXDEV
  cross-device os.replace in ensure_store_local); routed per workflow-fix-on-bug'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes`
failure-lesson raised on task #1335 (emitting agent: issue-session orchestrator,
r9 crash-fix round).

## Goal

Add a gotchas.md entry: a bare `tempfile.TemporaryDirectory()` (/tmp container
disk) + `os.replace` onto `/workspace` crashes EXDEV on pods — stage the
tempdir inside the destination dir.

## Workflow gap

- **Bug observed:** `issue1335_fit.py ensure_store_local` staged Hub shards in
  a /tmp tempdir and `os.replace`'d onto `/workspace`; crashed
  `OSError: [Errno 18] Invalid cross-device link` on pod-1335, killing the
  matched-n phase (attempt 7, 2026-07-16T12:42Z; recovered by the r9 hot-fix
  e74296e460).
- **Why it is a workflow gap:** `.claude/rules/gotchas.md` documents pod
  filesystem traps (MooseFS EDQUOT, FUSE read-wedge) but has NO entry for the
  /tmp-vs-/workspace cross-device `os.replace` trap, and an existing
  agent-memory recipe (`feedback_hf_local_dir_staging_for_delete_to_free.md`)
  actively RECOMMENDS the `hf_hub_download(local_dir=td)` + `os.replace`
  pattern without the same-filesystem caveat — so the trap will recur on any
  new download-then-move staging path. It works on same-device local dev and
  fails only on pods (latent until production).
- **Confidence (emitter):** high
- verified-at-filing: `grep -rn 'EXDEV|cross-device' .claude/rules/gotchas.md` → 0 hits
  (2026-07-16; absence-of-guard claim — the 0-hit in-target result IS the
  evidence that the gotcha is undocumented).

## Proposed change (candidate diff sketch — refine in planning)

```
+ ## os.replace across /tmp -> /workspace crashes EXDEV on pods
+
+ A bare tempfile.TemporaryDirectory() lives on /tmp (container disk); the
+ repo tree lives on /workspace (MooseFS/network volume). os.replace cannot
+ cross filesystems: OSError errno 18 (Invalid cross-device link). Stage
+ download-then-move paths in a tempdir INSIDE the destination dir —
+ TemporaryDirectory(dir=dest_dir, prefix=".hfstage_") — so the move stays
+ same-filesystem (atomic) and non-recursive resume globs cannot see the
+ half-downloaded nested tree (issue1335_extract_store.py precedent;
+ incident #1335 r9: ensure_store_local, attempt 7).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'TemporaryDirectory' .claude/ CLAUDE.md scripts/`) and consider
  whether the agent-memory recipe
  `.claude/agent-memory/experiment-implementer/feedback_hf_local_dir_staging_for_delete_to_free.md`
  should cross-reference the new entry (a complementary memory
  `feedback_exdev_tempdir_hub_staging.md` was already committed at
  d87e902661).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 23d8f7eb7e71

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: issue1335_fit.py ensure_store_local staged Hub shards in a /tmp tempdir and os.replace'd onto /workspace; crashed OSError errno 18 Invalid cross-device link on pod-1335, killing the matched-n phase
why_workflow_gap: gotchas.md documents pod filesystem traps but not the /tmp-vs-/workspace cross-device os.replace trap, and an agent-memory recipe recommends the local_dir + os.replace pattern without the same-filesystem caveat
proposed_change: Add a gotchas.md entry: a bare tempfile.TemporaryDirectory() (/tmp container disk) + os.replace onto /workspace crashes EXDEV on pods — stage the tempdir inside the destination dir
diff_sketch: |
  + ## os.replace across /tmp -> /workspace crashes EXDEV on pods
  + Stage download-then-move paths in TemporaryDirectory(dir=dest_dir,
  + prefix=".hfstage_") so os.replace stays same-filesystem (atomic) and
  + resume globs cannot see the half-downloaded tree.
confidence: high
related_task: #1335
<!-- /workflow-fix-candidate -->
