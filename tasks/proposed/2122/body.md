---
title: 'daily-fix: root-branch guard ssh/cd-form false positives'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c8d281f111ac
- daily-auto-filed
created_at: '2026-08-06T07:05:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): guard blocked ssh remote
  git payloads twice and the sanctioned /tmp scratch-worktree cd-form merge once'
workflow: v1
---
# daily-fix: guard_repo_root_branch.sh false positives — ssh remote payloads and /tmp scratch-worktree cd-form

## Workflow gap

`scripts/guard_repo_root_branch.sh` classifies by scanning command TEXT for git verbs, so
two sanctioned shapes are blocked as if they targeted the shared repo root:

1. **SSH remote payloads (2 independent firings).** 2026-08-05T15:40:56Z (#1739): the
   pod-side recovery `ssh <pod> "… git checkout --detach <sha>"` was blocked ("would move
   the SHARED repo-root tree off main") although the checkout targets the POD's clone;
   the agent had to rediscover the single-quoted recompose. 2026-08-06T02:42:52Z (#2091):
   `ssh pod-2091 '… git checkout issue-2091'` blocked the same way; recomposed with the
   `git -C` final-token form.
2. **Scratch-worktree cd-form.** 2026-08-05T14:13:41Z (#1491, Step 10d landing): the
   SANCTIONED scratch-worktree merge recipe was blocked because it was composed as
   `cd /tmp/i1491mrg && git merge …` — the guard does not recognize a leading
   `cd <non-root-path> &&` clause; the `git -C /tmp/i1491mrg merge` retry ran fine.

Each block cost only a turn, but the ssh shape fired twice in one day on two different
tasks and the guard's error text names neither sanctioned recompose.

verified-at-filing: all three firings are probed tool_result hook-error rows (sessions
2f4940f0 row 294, b765cdcd row 691, 8d7f8b25 row 4927). `grep -n 'ssh' scripts/guard_repo_root_branch.sh | head -5`
run at compose time — the guard has no ssh-payload carve-out today.

## Proposed change

Either teach the guard the two shapes (skip git verbs that appear only inside an
`ssh <host> '…'` remote payload — a remote cwd can never be the shared root; recognize a
leading `cd <path> &&` clause where <path> is outside the repo root before classifying),
or, if text-scanning cannot do that safely, extend the guard's BLOCK message to name the
sanctioned recomposes explicitly (single-quoted ssh payload / per-clause `git -C`), so
recovery is one turn instead of rediscovery. Keep the guard's conservative default —
false positives are acceptable; unnamed recovery shapes are the cost.

## Provenance

- fingerprint: c8d281f111ac

- workflow_fix_target: scripts/guard_repo_root_branch.sh
- origin: /daily 2026-08-05 problem sweep — miners 7 (P7), 6 (P16a), 1 (P13).
