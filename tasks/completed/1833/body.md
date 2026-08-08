---
title: 'workflow-fix: gotchas entry — out-arg FILE-vs-DIR kind; misnested deliverables
  pass .exists() checks'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1e51018bc728
created_at: '2026-07-29T16:50:32Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #1776 crash-fix cycle 5 (epm:failure-lesson,
  2026-07-29): out-arg file-vs-dir kind contract — lesson block in the v9 implementation
  marker'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson `gotcha_candidate: yes` block raised on task #1776 (emitting agent: experiment-implementer, crash-fix cycle 5).

## Goal

Add a `.claude/rules/gotchas.md` entry documenting the out-arg FILE-vs-DIRECTORY kind trap: a deliverable file path passed as a script's out-DIRECTORY arg misnests outputs one level deep while every `.exists()` / `[[ -e ]]` completeness check passes on the misnested shape — the crash surfaces a full phase later at upload.

## Workflow gap

- **Bug observed:** #1776's `dispatch.sh` passed the plan deliverable's FILE path as phase3's `--eval-out` DIRECTORY arg. `mkdir(parents=True)` created a directory NAMED like the file; the real outputs landed at `.../steered_shift_summaries.json/steered_shift_summaries.json`; the phase's own completeness assert (`(eval_out / "x.json").exists()`) passed BY COINCIDENCE of the nesting; the upload list's `[[ -e ]]` guard passed the directory; only `CommitOperationAdd`'s is_file check crashed — one full phase later, on 8×H100, costing a launch cycle. A sibling upload-list entry pointing at a never-written path was silently `-e`-skipped in the same block (the #825 class).
- **Why it is a workflow gap:** gotchas.md covers the upload-side file/folder API branch (`hub._upload` ValueError, ~L300, #595) but nothing documents the dispatcher-side out-ARG kind contract or the `.exists()`-satisfied-by-a-directory assert trap — dispatcher-invokes-phase-script is the standard experiment shape, so the class recurs.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n -i "is_file\|file-vs-dir\|file vs dir\|mkdir.*named" .claude/rules/gotchas.md` → 1 hit in 1 file (L300, the hub._upload per-file API branch — upload-side, distinct trap; NO entry covers the dispatcher out-arg kind / misnesting-passes-exists class; absence claim) (2026-07-29)

## Proposed change (candidate diff sketch — refine in planning)

```
+ - **An out-path arg's FILE-vs-DIRECTORY kind is part of the call contract —
+   and a misnested deliverable PASSES every .exists()/[[ -e ]] check.** A
+   dispatcher that passes a deliverable FILE path as a script's out-DIR arg
+   makes mkdir(parents=True) create a dir NAMED like the file; outputs land
+   one level deep (.../x.json/x.json); completeness asserts of the form
+   (out_dir/"x.json").exists() pass BY COINCIDENCE; [[ -e ]] upload-list
+   guards pass directories; the crash surfaces a phase later at
+   CommitOperationAdd (#1776 p3_upload, one 8xH100 launch cycle). Rules:
+   (i) check what the script DOES with each out arg (mkdir => dir;
+   open/write => file) when composing dispatcher invocations, and
+   class-sweep all phases' out args once; (ii) deliverable checks use
+   is_file()/[[ -f ]], never exists()/[[ -e ]]; (iii) scripts refuse
+   file-shaped out-dir args at argparse time; (iv) misnest repair branches
+   are exact-shape-guarded and RELOCATE, never delete. Sibling (upload-side
+   API kind): the hub._upload per-file entry below/above (#595). Long-form
+   twin: .claude/agent-memory/experiment-implementer/feedback_out_arg_file_vs_dir_kind.md
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Place near the hub._upload file/folder entry (~L300) so the kind-contract family clusters.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; gotchas.md `paths:` frontmatter untouched unless the trigger set genuinely widens.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 1e51018bc728
