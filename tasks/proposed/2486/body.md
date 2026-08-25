---
title: 'workflow-fix: Step 5a sibling-file sync commits silently after a ''no sync
  commit landed'' line, so briefs get a stale tip SHA'
kind: infra
tags:
- wf-fix
- workflow-fix
created_at: '2026-08-22T22:14:14Z'
has_clean_result: false
parent_id: 2263
origin_prompt: /issue 2263
workflow: v1
---
## Overview / Motivation

The Step 5a spec-freshness sync block has two independent arms that each may commit, but only one of them announces a commit — and the arm that stays silent runs immediately AFTER a line that reads as "HEAD did not move." An orchestrator that trusts the printed summary then composes reviewer briefs against a stale tip SHA.

Observed live during #2263 review round 8 (2026-08-22).

## Workflow gap

In `.claude/skills/issue/steps/09-step-5.md`:

- **Line 501** (family-sync arm): `echo "[step5a] no sync commit landed (no family drift, or the checkout errored above)"`
- **Line 587** (sibling-file arm): `echo "[step5a] sibling-file sync: ${#SIBLING_SYNCED[@]} file(s)"`

The sibling arm **does commit** when it syncs files, and it prints only a file COUNT — never the resulting SHA, and never any statement that HEAD advanced. Because the family message is emitted first, the combined output of a run with zero family drift and a non-empty sibling sync reads:

```
[step5a] no sync commit landed (no family drift, or the checkout errored above)
[step5a] sibling-file sync: 2 file(s)
```

The natural reading of that pair is "nothing was committed, two files were considered." The actual state is "a commit landed and HEAD moved."

## Demonstrated consequence (#2263 review round 8)

1. Pre-dispatch check recorded tip `abbd0481c7`.
2. Step 5a ran and printed exactly the two lines above.
3. Reading them as "HEAD unchanged," the orchestrator wrote tip `abbd0481c7` into BOTH reviewer briefs (Claude `code-reviewer` and the Codex composer).
4. The sibling arm had in fact committed `c90bc712f5` ("sync workflow-surface specs from origin/main (spec-freshness; sibling-issue files)", 2 added scripts).
5. The Codex composer independently re-derived HEAD, caught the staleness, pre-adjudicated the extra commits as excluded, and attested `git diff d1d8da42cc..HEAD -- <deliverable file>` was empty so live reads stayed safe.
6. The Claude reviewer received the stale tip figure with no correction.

Harmless **only** because the sibling sync happened to touch files the round did not. That is luck, not a property: sibling sync targets workflow-surface spec files, and rounds on this surface routinely edit spec files. A sibling-synced file that the round also touched would leave a reviewer reading a different tree than its brief describes — the exact class of divergence Step 5a exists to prevent.

## Why it is a workflow gap and not an orchestrator mistake

The orchestrator's only signal is the block's own summary output. The block knows it committed; it declines to say so. Correctness currently depends on the orchestrator independently re-running `git rev-parse HEAD` after every sync — which is precisely the redundant verification a summary line exists to make unnecessary. A reviewer-brief SHA is load-bearing (briefs pin the reviewable diff to explicit commits), so a silently-moved HEAD is a correctness surface, not cosmetics.

## Proposed change (sketch — refine in planning)

Three candidate levers; planning should pick, not necessarily all:

1. **Scope the family message** so it cannot be read as a global claim: `no FAMILY sync commit landed (…)` — it is already true of its own arm, just ambiguously worded.
2. **Have the sibling arm print its commit SHA** when it commits, not only a file count.
3. **Emit one terminal HEAD-state line** whenever ANY arm committed — e.g. `[step5a] HEAD advanced: <old> -> <new> (family=<n> sibling=<m>)`, and an explicit `[step5a] HEAD unchanged` otherwise. This is the version that makes the misread structurally impossible, and it is the one that gives the orchestrator a single line to key brief composition on.

Consider also whether Step 5's brief-composition step should assert its recorded tip equals live HEAD before dispatching reviewers — a cheap belt-and-braces check independent of whatever the sync prints.

## Dedup — distinct from three neighbours on the same block

- **#2302** (completed) — sibling sync puts main's own files in the branch diff. About synced CONTENT.
- **#2412** (completed) — satisfiability probe is `--collect-only`. About the PROBE.
- **#2423** (proposed) — satisfiability probe never runs on a SCRIPT-only sync. About the PROBE.

None concerns the arm's reporting or the silently-advanced HEAD. This is a distinct bug on a shared file.

## Coordination — probe before dispatching a writer

**#2423 is open against the same file** (`09-step-5.md`, sibling-sync region). Per `.claude/rules/cross-session-writer-arbitration.md`, run the pre-dispatch probe (`spawn_session.py list` + a `file-set claim:` marker scan + `git log --since` recency on the intended paths) and either sequence-after-commit or split to a disjoint region. #2263 also touches this file family while it is in flight.

## Verified at filing

- `grep -n` on `.claude/skills/issue/steps/09-step-5.md` confirms both echo lines at 501 and 587, with the sibling arm printing a count and no SHA (2026-08-22).
- `git log --oneline d1d8da42cc..HEAD` on `issue-2263` shows `c90bc712f5` between the deliverable and HEAD, authored by the Step 5a sibling arm.
- `git diff --stat d1d8da42cc..HEAD -- tests/test_verify_carryover_inputs.py` empty — why this instance was harmless.
- #2263 `events.jsonl`: the round-8 dispatch note records the stale tip; the Codex composer's return records the catch.

## Provenance

workflow_fix_target: .claude/skills/issue/steps/09-step-5.md

Routed from a defect the #2263 orchestrator hit in its own Step 5a run, per `.claude/rules/workflow-fix-on-bug.md`. `task_workflow.is_workflow_fix_session(2263)` is `False`, so the recursion guard does not apply, and the target region is unrelated to #2263's deliverable.
