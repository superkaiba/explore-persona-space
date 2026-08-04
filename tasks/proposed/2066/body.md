---
title: 'workflow-fix: root-commit guard is cwd-blind — worktree comm'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9df4f5fdd783
- daily-auto-filed
created_at: '2026-08-04T06:52:39Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-03 problem sweep (route 2): guard_root_code_commit.sh
  blocked >=7 commits across 5 unrelated sessions on 2026-08-03/04, listing the same
  6 foreign untracked root drafts as the uncertified payload, including commits whose
  cwd was provably a worktree; the hook header documents the CWD-BLIND Layer-2 root-index
  read (L62-65) and buries the git -C remediation in a comment.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-03 (route 2: behavior/logic change → independent review) from the nightly problem sweep (miners 4/5/6/8 — four independent observations across five sessions).

**Related open sibling: #2046** (`daily-fix: root-commit guard honors excluding pathspec`) targets the SAME hook for the adjacent ROOT-cwd pathspec case. Distinct fingerprint, and the two fixes touch the same block-message / scoping code — the planner should consider landing them together.

## Goal

A commit issued from a linked worktree must not be blocked by uncertified files sitting in the SHARED ROOT's index, and when the guard does block conservatively its message must lead with the remediation instead of burying it in the script's comments.

## Workflow gap

- **Bug observed:** `guard_root_code_commit.sh` blocked at least 7 commits across 5 unrelated sessions on 2026-08-03/04, each listing the same foreign untracked root drafts (`scripts/issue1482_*.py` ×5, `scripts/issue1895_overlap_ksweep_fig.py`) as the "UNCERTIFIED code payload" — including commits whose cwd was provably a worktree (session 52bc7fdf probed `pwd` → `.../worktrees/issue-2051` between two blocks) and one (3654b1da, 23:51:38Z) whose block listed ONLY the foreign files after the session's own payload was already lint-certified. Firing events (deduped per tool call): 3654b1da ×3 (23:33:09/23:42:27/23:51:38Z), 52bc7fdf ×2 (23:26:33/23:26:54Z), 395ac452 ×1 (03:58:59Z), cd57c423 ×2 (21:50:56/22:02:24Z), 31554c9f ×1 (04:36:27Z), 6a2f91cc ×1 (01:24:26Z). Each cost 2–4 turns rediscovering the `git -C "$WT"` form.
- **Why it is a workflow gap:** the hook's own header documents this as a known limitation — "CWD-BLIND (pull-guard parity): a bare `git commit` issued while the Bash shell's inherited cwd is a worktree matches Layer 1, but Layer 2 reads the ROOT's index — it allows unless the root simultaneously has gated files staged. Remediation: `git -C "$WT" commit`" (lines 62–65), and "pathspec SCOPING engages only when the hook-input cwd provably equals the root ... non-root-cwd commits stay conservatively [blocked]" (lines 68–71). On a normal day the escape hatch holds because the root index is clean; today the root carried 6 foreign drafts, so the documented "allows unless" condition inverted and the limitation became a fleet-wide tax. The remediation exists only in a comment — the block message does not lead with it.
- **Confidence (emitter):** high (the mechanism is quoted from the hook's own docstring; the firing counts are transcript-grounded).
- verified-at-filing: `sed -n '62,71p' .claude/hooks/guard_root_code_commit.sh` → the CWD-BLIND and pathspec-scoping paragraphs quoted above, verbatim in-target (2026-08-04). `git status --short | grep -E 'issue1482|issue1895'` → **6** `??` untracked hits still present at the repo root now, i.e. the triggering condition is live and will keep firing. `grep -n 'cwd' .claude/hooks/guard_root_code_commit.sh` → the cwd gate at L351 (`*.claude/worktrees/*) cd_verdict=latch`) shows a worktree-cwd verdict already exists in Layer 1.
- unverified hypothesis — verify at plan time: whether extending Layer-2 scoping to a provably-worktree cwd is safe. The header calls the current direction "conservatively" safe; loosening it is the risk-bearing half of this fix, and the message-first half (below) is available with no safety change at all.

## Proposed change (candidate sketch — refine in planning)

Two legs; the SECOND is safe on its own and the planner may land only that:

```
(a) scope the Layer-2 payload read to the invoking worktree when the hook-input
    cwd is provably a linked worktree (parity with the #1620 root-cwd pathspec
    scoping already implemented)

(b) have the block message LEAD with the concrete rewrite of the blocked
    command — `git -C "$WT" commit -F <file> -- <paths>` — rather than leaving
    it in the script header; today every blocked session rediscovers it
```

## Scope / surfaces

- Primary target: `.claude/hooks/guard_root_code_commit.sh`.
- Cross-reference #2046 (same file, pathspec case) before editing; land together if the planner judges the block-message rework shared.

## Constraints / invariants

- The guard's block direction must stay fail-safe: any ambiguity (unknown cwd, quoted/spacey pathspecs) keeps blocking.
- Leg (a) must not create a path by which an uncertified ROOT payload commits from a worktree cwd.
- `scripts/workflow_lint.py` (no-flags) passes; `bash -n` on the hook passes; the hook's own test coverage (if any) stays green.

## Provenance

- sha-verify (filing-time, #1467): `52bc7fdf` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `3654b1da` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `395ac452` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `cd57c423` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `31554c9f` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `6a2f91cc` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: 9df4f5fdd783

- workflow_fix_target: .claude/hooks/guard_root_code_commit.sh
