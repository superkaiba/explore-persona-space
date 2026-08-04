---
title: 'workflow-fix: blanket-add latch misses equivalent-blanket sp'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fb9407e72b64
- daily-auto-filed
- trigger-dense
created_at: '2026-08-02T07:04:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): Pre-existing (outside #1977''s
  scope): the blanket-add latch arm matches only the exact tokens -A/--all/., so equivalent-blanket
  spellings such as `./` (and shell-expanding `*`, and `:/` magic) chained to a root
  commit bypass the latch entirely; the pre-add staged read + literal-token text_paths
  then miss everything the add stages, so gated content can land uncertified.'
workflow: v1
---
# workflow-fix: blanket-add latch misses equivalent-blanket spellings

## Overview / Motivation

Auto-filed by the /daily 2026-08-01 Step C parked-candidate sweep from a workflow-fix candidate parked on task #1977 (emitting agent: critic, plan-review round 1, recursion-guarded; formal candidate block, fingerprint d4f2b68c9bb8).

## Goal

Extend `guard_root_code_commit.sh`'s blanket-add latch (and its B13 test parametrization) to the equivalent-blanket token spellings (`./`, `:/`, and a conservative treatment of bare `*`), or normalize candidate tokens before the exact match, so gated content cannot land uncertified via an alternate spelling.

## Workflow gap

- **Bug observed:** the blanket-add latch arm matches only the exact tokens `-A`/`--all`/`.`, so equivalent-blanket spellings such as `./` (and shell-expanding `*`, and `:/` magic) chained to a root commit bypass the latch entirely; the pre-add staged read + literal-token text_paths then miss everything the add stages, so gated content can land uncertified. (Pre-existing; flagged as outside #1977's scope by its plan critic.)
- **Why it is a workflow gap:** the guard's blanket-staging block enumerates 3 exact token spellings while git accepts several equivalent blanket spellings; the hook is the workflow surface enforcing the CLAUDE.md shared-root staging ban.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'a_saw_blanket' .claude/hooks/guard_root_code_commit.sh` → recognition arm at L811: `case "$tok" in -A | --all | .) a_saw_blanket=1 ;; esac` — exactly the 3 exact tokens, no `./`/`:/`/`*` handling, re-confirmed AFTER #1977's own merge landed (`792920685d`, "exempt path-limited git add --all", 2026-08-01) reshaped the latch into the `a_saw_blanket`/`a_eligible` form without widening the recognition set (2026-08-02 UTC). Note the emitter's cited line 678 has drifted to L811 post-#1977; the candidate's `add:-A | add:--all | add:.` sketch reflects the pre-#1977 shape — the planner should target the current L806-856 block.

## Proposed change (candidate diff sketch — refine in planning; sketch predates the #1977 reshape)

```diff
- case "$tok" in -A | --all | .) a_saw_blanket=1 ;; esac
+ case "$tok" in -A | --all | . | ./ | .// | :/) a_saw_blanket=1 ;; esac
  (+ conservative treatment of bare `*`; + B13 test ids for the new spellings)
```

## Scope / surfaces

- Primary target: `.claude/hooks/guard_root_code_commit.sh`, `tests/test_guard_root_code_commit.py`
- Coordinate with the just-landed #1977 path-limited `git add --all -- <pathspec>` exemption — the widened recognition must not re-block the sanctioned shape.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; `bash -n` on the hook passes; the B13 parametrized tests cover every new spelling.
- Recursion guard applies (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: .claude/hooks/guard_root_code_commit.sh, tests/test_guard_root_code_commit.py
- fingerprint: fb9407e72b64 (tag-authoritative; supersedes body-carried fingerprint: d4f2b68c9bb8)
- origin: parked candidate on task #1977, ts 2026-08-01T07:31:06Z, routed by /daily 2026-08-01 Step C.

<!-- workflow-fix-candidate v1 -->
target_file: .claude/hooks/guard_root_code_commit.sh, tests/test_guard_root_code_commit.py
bug_observed: Pre-existing (outside #1977's scope): the blanket-add latch arm (guard_root_code_commit.sh:678) matches only the exact tokens -A/--all/., so equivalent-blanket spellings such as `./` (and shell-expanding `*`, and `:/` magic) chained to a root commit bypass the latch entirely; the pre-add staged read + literal-token text_paths then miss everything the add stages, so gated content can land uncertified.
why_workflow_gap: The guard's blanket-staging block enumerates 3 exact token spellings while git accepts several equivalent blanket spellings; the hook is the workflow surface enforcing the CLAUDE.md shared-root staging ban.
proposed_change: Extend the latch arm (and B13 parametrization) to the equivalent-blanket token spellings (`./`, `:/`, and a conservative treatment of bare `*`), or normalize candidate tokens before the exact match.
confidence: medium
related_task: #1977
<!-- /workflow-fix-candidate -->
