---
title: 'workflow-fix: gotchas — stage_hub_prefix dest is a mirror root (caller layout
  trap)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5d3fc3733933
created_at: '2026-07-29T03:57:39Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate failure-lesson from task #1774 crash-fix round (att-20260729-033609);
  see the epm:failure-lesson v1 marker on task 1774'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes`
failure-lesson raised on task #1774 (emitting agent: issue-1774 orchestrator,
crash-fix round for GCE attempt att-20260729-033609).

## Goal

Add a gotchas.md entry for the `stage_hub_prefix` caller-side layout trap: its
`dest_dir` is a MIRROR ROOT (files land at `dest/<repo-relative path>`), so passing
the FINAL consumed path as dest nests the whole hub prefix under it — invisible until
the staging path first runs on a machine where the store does not pre-exist.

## Workflow gap

- **Bug observed:** #1774's P0 restage passed the consumed path as `dest_dir`; on the
  fresh GCE clone the restage "succeeded" but the files landed at
  `corpus/issue1092_realistic_crossing/corpus/manifest.jsonl`, the post-restage check
  re-raised FileNotFoundError, and a GCE launch cycle was burned (exit-trap DELETE).
- **Why it is a workflow gap:** gotchas.md's staging entry names `hub.stage_hub_prefix`
  as the canonical helper but documents only the enumeration/quota side — the
  dest-layout contract (verbatim prefix mirror) and the (h)(iv) consumer-open probe
  duty are not stated where implementers read them; the trap recurs for any new
  staging caller (two independent call sites in #1774 alone had it).
- **Confidence (emitter):** high
- verified-at-filing: `grep -c "stage_hub_prefix" .claude/rules/gotchas.md` → 1 hit
  (the L334 staging-recipe entry naming the helper as canonical; its context documents
  enumeration/quota — read, does NOT implement the proposed dest-layout caveat)
  (2026-07-29)

## Proposed change (candidate diff sketch — refine in planning)

```
+ - **`stage_hub_prefix`'s dest is a MIRROR ROOT — files land at
+   `dest/<repo-relative path>` (verbatim prefix mirror).** Passing the FINAL
+   consumed path as dest nests the hub prefix under it; invisible until the
+   staging path first runs on a fresh clone (#1774 att-20260729-033609: P0
+   restage "succeeded", post-restage check re-raised FileNotFoundError, one
+   GCE cycle burned). Pass a root satisfying root/<hub prefix> == consumed
+   path (assert the arithmetic), or stage into a scratch mirror + rename the
+   leaf; probe with one REAL small-prefix staging + the consumer's own open
+   (artifact-reuse check (h)(iv)) before production. stage_hub_file (exact
+   dest per file) does not have this trap. Worked fix:
+   6948bdd1fe60c73b63e10b1ec04e633083c91c63 + tests/test_issue1774_round3.py;
+   memory: feedback_stage_hub_prefix_dest_is_mirror_root.md.
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'stage_hub_prefix' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan. (Consider a one-line cross-reference in
  `.claude/rules/artifact-reuse.md` check (h)(iv) and `upload-policy.md`'s
  staging-download paragraph, which already documents the mirror in passing.)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 5d3fc3733933

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: P0 stage-audit restage (issue1774_stage_audit.py verify_or_stage_store)
lesson: hub.stage_hub_prefix lands files at dest/<repo-relative path> (verbatim prefix mirror) — passing the FINAL consumed path as dest nests the whole hub prefix under it, and the bug is invisible until the restage path first runs on a machine where the store does NOT pre-exist (fresh clone). Pass a mirror ROOT satisfying root/<hub prefix> == consumed path, and probe the mapping with a 1-file real staging + consumer-open before production (artifact-reuse check (h)(iv)).
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
supersedes:
<!-- /epm:failure-lesson -->
