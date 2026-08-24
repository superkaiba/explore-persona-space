---
title: 'workflow-fix: Step 5a spec-freshness SPECS omits .gitleaksignore — the sibling-file
  arm stages sibling scripts it cannot commit'
kind: infra
tags:
- wf-fix
created_at: '2026-08-24T20:20:35Z'
has_clean_result: false
origin_prompt: 'Hit during #2537''s Step 9c pre-gate re-sync: the sibling-issue arm
  imported three issue2215_* scripts from main, gitleaks refused the commit on a fingerprint
  main had already vetted at .gitleaksignore:1450, and the worktree''s copy lacked
  it because .gitleaksignore is not in the sync''s SPECS set.'
workflow: v1
---
## Goal

Add `.gitleaksignore` to the Step 5a spec-freshness sync's file set, coupled to the sibling-issue
script/test arm (#1972). Today the arm imports sibling scripts from `main` while leaving behind the
ignore file whose entries make those scripts committable, so the sync stages files it cannot commit.

## Evidence

Hit during #2537's Step 9c pre-gate re-sync (2026-08-24). Reproduced end to end:

1. The Step 5a block's sibling-issue arm ran `git checkout origin/main -- <paths>` for three files
   added to `main` after this branch's cut: `scripts/issue2215_separation_comparison.py`,
   `scripts/issue2215_sepcmp_cell_examples.py`, `scripts/issue2215_sepcmp_qwen_embed.py`.
2. Its commit was refused by the `gitleaks` pre-commit hook:
   `Fingerprint: scripts/issue2215_sepcmp_qwen_embed.py:generic-api-key:11`, entropy 3.70.
3. The finding is a **false positive** — line 11 is docstring prose naming a model,
   ``Qwen/Qwen3-Embedding-8B`` via the vLLM pooling runner``, not a credential.
4. `main` had ALREADY vetted it: `.gitleaksignore` on `origin/main` carries that exact fingerprint at
   line 1450.
5. The worktree's `.gitleaksignore` did NOT (1449 lines vs main's 1450), because `.gitleaksignore`
   appears **nowhere** in the Step 5a `SPECS` list (`grep -c gitleaksignore` over
   `.claude/skills/issue/steps/09-step-5.md` → 0).

Net effect: the sync left three files staged-but-uncommitted in the worktree, and the round had to
hand-sync `.gitleaksignore` from `origin/main` to clear it (#2537 commit `5f2e470808`).

## Why this is a coupling, not a missing singleton

The Step 5a block already models coupled families explicitly — `FAMILY_workflow`, `FAMILY_lint`,
`FAMILY_guard`, `FAMILY_agents` — on exactly this principle: syncing one member without its partner
manufactures vintage skew. `.gitleaksignore` is the same shape with respect to the sibling-file arm:
the arm's imported content is only committable against main's version of the ignore file. A sibling
script whose fingerprint was vetted on `main` is un-committable in any worktree cut before that
vetting landed.

It is also strictly a fail-CLOSED skew (a refused commit, loud), never a clobber — which is why this
is a workflow-ergonomics fix rather than a correctness incident.

## Scope

1. Add `.gitleaksignore` to `SPECS` in the Step 5a spec-freshness block
   (`.claude/skills/issue/steps/09-step-5.md`), and decide its family membership deliberately:
   either couple it to the sibling-file arm, or make it a singleton that syncs whenever it is itself
   clean. State the reasoning in the block's comment the way the existing families do.
2. Check the drift guard. `tests/test_issue_skill_lint_family_sync.py` guard (20) pins family
   completeness, and `test_step10d_family_atomicity_matches_step5a` pins that Step 9c/10d do not
   inline a third `FAMILY_OF` copy — confirm the addition satisfies both rather than tripping them.
3. Confirm the branch-side-edit guard still protects a branch that legitimately edits
   `.gitleaksignore` itself (a round adding its own vetted fingerprint must not have it clobbered).

## Acceptance

1. A worktree cut before a sibling script's `.gitleaksignore` vetting can run the Step 5a sync and
   commit the imported sibling files with no hand intervention.
2. The existing family-sync pin tests stay green.
3. A branch that has itself modified `.gitleaksignore` is skipped per the per-item branch-side-edit
   guard (fail-safe: status-quo staleness, never a clobber).

## Notes

Low urgency, high annoyance: it costs a hand-sync per affected round rather than corrupting anything.
Found by #2537, which is itself a Step 9c selector-mapping fix — the two are independent gaps in the
same gate's plumbing.
