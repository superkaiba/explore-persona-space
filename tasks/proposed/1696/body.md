---
title: 'daily-fix: guard-surface reviewer posts verdict first'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a4f81976404b
- daily-auto-filed
created_at: '2026-07-26T07:05:32Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): Four spurious usage-policy
  refusals hit one nine-session wave, three of them killing a reviewer at its final
  summarization turn after 15 to 24 tool calls so a completed verdict was destroyed
  and had to be re-earned, and a clarifier whole-file read of a guard script was hard-blocked
  by the trigger-dense read guard.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 problem sweep. Four spurious usage-policy
refusals hit the #1667–#1689 wave; three killed a reviewer at its FINAL turn, after
all the review work was already done.

## Goal

On a guard/hook-surface round, require the reviewer to post its verdict marker via
`task.py post-marker --file` BEFORE composing its final in-context report, and tell
the clarifier to read `.claude/hooks/` targets windowed rather than whole-file.

## Workflow gap

- **Bug observed:** four refusals across nine sessions, all on guard/hook-surface
  tasks (piped-git guard, root-code-commit guard, dedup tokens):
  - #1673 (`8e1e40cf`) @ 07:21:35Z — consistency-checker killed.
  - #1675 (`5cfcf606`) @ 12:29:57Z — code-reviewer killed **15 tool calls in**, at its
    final turn. Session marker: *"first reviewer spawn refusal-killed at its final
    turn (spurious usage-policy refusal, 15 tool calls, no durable verdict — Step 5b
    durable-verdict-first check confirmed no epm:code-review marker); re-spawned
    ONCE…"*
  - #1676 (`30513980`) @ 12:41:59Z — code-reviewer killed **24 tool calls in**.
  - #1687 (`b656f7fa`) @ 07:54:25Z — the orchestrator turn itself.
  Cost ≈2 min + ≈6 min + ≈15 min of *completed but discarded* review work, plus
  respawn latency.
- **Why it is a workflow gap:** `.claude/rules/trigger-dense-review.md` and the
  #1503/#1413 first-pass-brief discipline already neutralize what the reviewer
  **reads**. Three of these four kills fired on what the reviewer **writes** — its own
  final summarization turn, quoting the guard shapes it just reviewed. The sessions
  pre-materialized trigger-dense excerpt files as the rules require and were killed
  anyway. Ordering is the lever the existing rules do not pull: the verdict marker is
  the durable artifact, and it is currently composed *after* the report that gets
  killed.
- **Second, smaller item (same rule family):** #1676 (`30513980`) @ 11:15:54Z — the
  clarifier attempted a whole-file `Read` of `.claude/hooks/guard_root_code_commit.sh`
  and `guard_trigger_dense_read.sh` blocked it (*"BLOCKED: unbounded Read of
  …guard_root_code_commit.sh … Read it WINDOWED instead"*). The hook fired correctly;
  the gap is that the Step 1 clarifier context-gathering block still leads with an
  unbounded read for hook targets.
- **Confidence (emitter):** high on the pattern (3 of 4 kills at the final turn);
  medium on the remedy's completeness — posting the marker first bounds the LOSS but
  does not prevent the refusal.
- verified-at-filing: absence confirmed in the named target —
  `grep -c 'verdict.*before.*report\|post-marker.*before' .claude/rules/trigger-dense-review.md`
  → **0**; the rule's existing sections govern reviewing/reconciling guard artifacts,
  composing briefs on such targets, orchestrator ingest of run-failure text, and
  orchestrator turns on a guard-surface round (per the LESSONS.md index row) — all
  READ-side and brief-side, none ordering the reviewer's own output. Refusal counts
  cross-checked against the transcripts: exactly **1** `isApiErrorMessage: true`
  assistant row across the nine `/issue` transcripts of that miner's slice (the
  orchestrator kill); the three subagent kills are counted from the orchestrator's
  recorded recovery markers, since a killed subagent's rows live in its own transcript.
  Landed-fix history check `git log --oneline --since='7 days ago' --
  .claude/rules/trigger-dense-review.md` → no commits. (2026-07-25)

## Proposed change (refine in planning)

```
  .claude/rules/trigger-dense-review.md — new clause, guard/hook-surface rounds:
+ the reviewer's FIRST durable action after reaching a verdict is
+   uv run python scripts/task.py post-marker <N> epm:code-review --file <path>
+ composing the in-context report only AFTER the marker lands, so a final-turn
+ refusal costs the report (recoverable) rather than the verdict (a full respawn).
+ Pairs with the Step 5b durable-verdict-first check, which already treats a
+ present marker as the authority.

  .claude/skills/issue/SKILL.md — Step 1 clarifier context-gathering:
+ when the task target is under .claude/hooks/, read it WINDOWED
+ (grep-anchored offset/limit), never whole-file.
```

## Scope / surfaces

- Primary target: `.claude/rules/trigger-dense-review.md`.
- `.claude/skills/issue/SKILL.md` Step 1 (the windowed-read sentence).
- Check whether `.claude/agents/code-reviewer.md` needs a matching pointer so the
  reviewer sees the ordering rule in its own spec, not only in the rule file.
- Marker-shape compatibility matters: the marker must still satisfy the Step 5c-bis
  mechanical-contract strip, so post the SAME body it would have posted, just earlier.

## Constraints / invariants

- Do not weaken review quality to dodge refusals: the reviewer still writes the full
  report; only the ORDER changes.
- The marker body itself is trigger-dense by nature — `--file` (never `--note`) is the
  required form, which the repo-root guard already forces for such text.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/trigger-dense-review.md
- fingerprint: a4f81976404b
- Source: `/daily` 2026-07-25 transcript sweep, sessions `8e1e40cf` (#1673),
  `5cfcf606` (#1675), `30513980` (#1676), `b656f7fa` (#1687).
