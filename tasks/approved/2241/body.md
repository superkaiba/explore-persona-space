---
title: 'Step 4a: draft-PR creation is gated on a condition that is false by construction
  (generator of the #2240 zero-PR class)'
kind: infra
tags:
- step4a-draft-pr-timing
created_at: '2026-08-12T05:57:29Z'
has_clean_result: false
origin_prompt: '/issue 2240 (Step 10b companion filing: plan v5 scope decision 3 —
  #2240 fixed the Step 10d recovery, this is the Step 4a generator)'
workflow: v1
---
# Step 4a: draft-PR creation is gated on a condition that is false by construction

## The gap

`/issue` Step 4a (`.claude/skills/issue/SKILL.md`, the `gh pr create --draft` block) opens the draft PR only when the branch is ahead of `origin/main`:

```bash
if [ "$(git -C "$REPO_ROOT" rev-list --count origin/main..issue-<N>)" -gt 0 ]; then
  gh pr create --draft --head issue-<N> ...
else
  echo "issue-<N> has no commits ahead of origin/main yet; skipping draft PR ..."
fi
```

But **Step 4a runs at the `approved`→`running` transition — before Step 4b dispatches the implementer**, so a freshly created branch has zero commits and the else arm fires *by construction*. No later step re-runs the create: the only other `gh pr create` in the skill is the Step 10d fresh-PR arm. The result is a standing population of code-bearing branches with no PR object.

Until #2240 landed, such a branch was then **silently skipped** by Step 10d's unconditional `-z "$PR"` arm and left permanently unmerged with the durable record reading clean (the #456→#466 stranded-shared-module class). #2240 fixed the *recovery* — Step 10d now creates the PR and merges — but deliberately left the *generator* in place as out of scope, because moving PR creation touches the hot path of every round.

## Measured incidence

Sampled 8 recent issue branches on 2026-08-12: **3 of 8 had zero PRs** — `issue-1739` (37 commits ahead, live session), `issue-2239` (1 commit ahead, live session), `issue-2117` (branch not even pushed to origin). `issue-2240`'s own branch reproduced it during its own run, and needed the by-hand fixed procedure to land.

The `[step10d-no-pr-anomaly]` marker #2240 added is the incidence counter for this task: every firing is one branch that hit the generator.

## Goal

Make draft-PR creation happen at a point where it can actually succeed, so Step 10d's recovery arm becomes a backstop rather than the normal path.

## Candidate approaches (not yet chosen — this needs a plan)

1. **Create on first push.** After the implementer's first round-push (Step 5), create the draft PR if none exists. Closest to the true fix; costs a `gh pr view` (or a cached check) on the round-push path.
2. **Make Step 4a idempotent and re-invoke it** at the first point where commits exist.
3. **Leave creation where it is but drop the commits-ahead gate** — REJECTED as structurally impossible: GitHub refuses a PR with zero commits between base and head, which is exactly why the gate exists. Recorded so a future planner does not re-derive it.

Approach 1 is the presumptive design; the plan should price the hot-path cost and decide whether the check is per-round or once-per-branch.

## Acceptance

- A branch that receives its first commit during a normal `/issue` run has an open draft PR before it reaches Step 10d.
- The `[step10d-no-pr-anomaly]` marker stops firing for branches that ran the full pipeline (it should remain reachable for genuinely abnormal cases).
- No new failure surface on the round-push hot path: a PR-create failure must not break or delay the round.
- A test pins whichever step becomes responsible for creation.

## Provenance

Filed by the `/issue 2240` session at Step 10b as the recorded companion to plan v5 scope decision 3. #2240 is the recovery layer; this task is the generator. Dedup key: same `target_file` (`.claude/skills/issue/SKILL.md`), DISTINCT bug — #2240 fixed the merge-time skip, this fixes the creation-time miss.

The Alternatives-lens critic on #2240's plan review specifically flagged that leaving this as plan prose would lose it, and that the marker's forensic value is as this task's incidence counter.

**Not auto-spawned:** the filing session was functionally a workflow-fix session, so the recursion guard was honoured as ON — exactly one distinct-root-cause task filed, left at `proposed` for the user/PM to dispatch rather than self-spawning.
