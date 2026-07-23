---
title: 'daily-fix: lint bare list_repo_files on the data repo'
kind: infra
tags:
- wf-fix
- wf-fix-fp:20565fe587b3
- daily-auto-filed
created_at: '2026-07-23T07:01:00Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): two fresh verify code paths
  called bare list_repo_files on the ~1M-file data repo and wedged (>7 min, killed)
  on 2026-07-22 despite the documented scoped-listing rule — no mechanical gate exists'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript sweep). Two freshly written verification code paths called bare `list_repo_files` on the ~1M-file data repo (`superkaiba1/explore-persona-space-data`) and wedged: (a) #779's `random_direction_null.py` post-upload verify hung >7 min and had to be pkill'ed (exit 144), replaced by a targeted probe that "returns True instantly"; (b) two background HF-listing probes in f4b1d707 hung and were killed. The RULE already exists (upload-policy.md line ~260: "a bare `list_repo_files` full listing of the ~1M-file data repo times out (>90 s, #833)"; the canonical verify helper `hub.verify_repo_paths_uploaded` is server-side scoped) — what is missing is the MECHANICAL gate: nothing flags a new script that writes the bare call.

## Goal

`scripts/workflow_lint.py` gains a check flagging a bare `list_repo_files(` call targeting the data repo (or with no `path_in_repo`/prefix scoping) in scripts/ + src/, pointing at the scoped `list_repo_tree` / `verify_repo_paths_uploaded` recipes; existing legitimate sites grandfathered via the lint's allowlist convention.

## Workflow gap

- **Bug observed:** two wedge events in one day (fdf687f2 00:02–00:09Z; f4b1d707 ~10:06Z) from freshly written code violating the documented rule — the doc alone is not preventing new bare-call sites.
- **Why it is a workflow gap:** the repo is now large enough that ANY full listing is a wedge; this is the recurring "rule exists, mechanical gate missing" class the lint exists for.
- **Confidence:** medium-high.
- verified-at-filing: `grep -n 'list_repo_files' .claude/rules/upload-policy.md` → guidance present at lines 260 (scoped list_repo_tree prescription, bare call times out) and 421 (the "fresh list_repo_files listing" there refers to the SCOPED canonical helper `hub.verify_repo_paths_uploaded`, #997 — context read, not a contradiction); `grep -rn 'list_repo_files' scripts/workflow_lint.py` → no existing check (absence claim), 2026-07-23 UTC.

## Proposed change (refine in planning)

Add `--check-bare-list-repo-files` (bundled into the no-flags run per lint convention): flag `list_repo_files(` sites in `scripts/` + `src/` that reference the data-repo id (or pass no scoping) with an allowlist for the existing legitimate sites; message names the scoped recipes.

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py` (+ its tests).

## Constraints / invariants

- No behavior change to library code; lint-only. `test_live_trees_pass`-style invariant: the check passes on the current tree after grandfathering. Recursion guard applies.

## Provenance

- fingerprint: 20565fe587b3

- workflow_fix_target: scripts/workflow_lint.py
