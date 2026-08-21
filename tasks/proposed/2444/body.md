---
title: 'Step 5a/10d spec-freshness: dirty-family scan is fail-open for a deliverable
  commit using the sync subject (clobbers agent-memory)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-21T07:51:46Z'
has_clean_result: false
parent_id: 2246
origin_prompt: 'Found during #2246 round 1 Step 5a sync (2026-08-21): commit 152b68ab4e
  carried agent-authored memory additions under the canonical spec-freshness subject,
  the pass-1 exclusion scored .claude/agent-memory CLEAN, and the surgical refresh
  reverted 79 lines. Distinct from #1789 (false-positive direction).'
workflow: v1
---
## Overview / Motivation

The Step 5a / Step 10d spec-freshness **dirty-family scan is fail-OPEN when a deliverable commit uses the exact prescribed sync-commit subject**: the family is scored CLEAN and the surgical `origin/main` refresh silently reverts the branch-side content.

Observed live during #2246 round 1 (2026-08-21), which lost 79 lines of agent-authored memory content until it was manually restored.

## The mechanism

`.claude/skills/issue/steps/09-step-5.md` pass-1 computes branch-side commits per spec path and filters them:

```
git log --format='%H %s' "$MB"..HEAD -- "$f" | awk 'index($0, "sync workflow-surface specs from") == 0'
```

Any commit whose subject contains that anchor phrase is dropped from the branch-side set. The intent is that a family's OWN prior sync commits must not poison later freshness checks on the same branch. The hole: the filter keys on the SUBJECT alone and never inspects the commit's CONTENT, so a commit that carries genuine deliverable content **under** that subject is indistinguishable from a real sync commit.

When that happens the family is marked clean, pass 2 admits it to `SAFE_SPECS`, and `git checkout origin/main -- $SAFE_SPECS` reverts the deliverable to main's copy. `.claude/agent-memory` is the highest-exposure member: it is a SINGLETON family (no coupling), so nothing else can mark it dirty, and its files are append-only, so a revert is a pure content loss with no conflict signal.

## This is NOT #1789 (distinct bug, same file)

#1789 (`completed`) fixed the **false-positive** direction: a deliverable commit whose subject merely *contained* the bare token `spec-freshness` was wrongly excluded, so it tightened the match from the bare token to the full sync-subject SHAPE.

This task is the **false-negative** converse: the subject genuinely IS the prescribed shape. #1789's tightening cannot catch it by construction — tightening the pattern only ever shrinks the excluded set, and this commit is inside it however tightly the shape is specified. Per the dedup rule, a distinct bug on the same file files its own task.

## Reproduction (from #2246, all shas on `issue-2246`)

1. `152b68ab4e` — subject `issue-2246: sync workflow-surface specs from origin/main (spec-freshness)`, content: `M` on two agent-memory files, +33 lines `codex-code-reviewer/feedback_revision_round_compose_recipe.md`, +49 lines `experiment-implementer/feedback_model_venv_pin_full_dep_closure_flashinfer.md`. Both are agent-authored memory entries (the codex one cites "#2214 r2, 2026-08-20"); neither exists on `origin/main`.
2. Step 5a ran and printed dirty-family skips for `workflow` and `lint` (this round's real edits) but NOT for `.claude/agent-memory`.
3. Sync commit `69b44089a9` reverted both files to main's content: `-30` / `-49`.
4. Verification that the loss was real, not a legitimate main-side update: for BOTH files `origin/main`'s blob is byte-identical to `152b68ab4e^`, so `152b`'s blob was exactly main's content plus the appended entry.
5. Manually restored in `80ce3f3fd3`.

## Acceptance criteria

- A branch-side commit that touches a spec path and carries content NOT present at `origin/main` marks its family dirty, **regardless of subject**.
- A genuine sync commit (content byte-identical to the `origin/main` blobs it names) still does NOT mark the family dirty — the #1789/#1560 no-poison property is preserved.
- A regression test covers both directions: (a) deliverable-content-under-sync-subject ⇒ family DIRTY; (b) true-sync-content-under-sync-subject ⇒ family CLEAN.
- The same fix applies to the Step 10d post-gate re-sync, which shares this scan.

## Suggested direction (not binding on the implementer)

Replace the subject-only filter with a **content** test: for each candidate commit, treat it as a sync commit only if every spec path it touches is byte-identical to the corresponding `origin/main` blob at scan time (`git rev-parse <commit>:<path>` vs `origin/main:<path>`). Subject may stay as a cheap pre-filter, but it must not be the decider. A content test is the only form that distinguishes the two cases, since the subject is identical by construction.

A cheaper partial mitigation (worth considering as defence in depth, not a substitute): make `.claude/agent-memory` never blind-syncable, since it is append-only and a revert there is always a loss.

## Provenance

Found while #2246 (Step 10d reap-shield) ran its own Step 5a sync before the round-1 code-review dispatch. #2246's round is unaffected after the restore; this task carries the general fix.
