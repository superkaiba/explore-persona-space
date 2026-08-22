---
title: 'workflow-fix: condensed launch references in 12-step-8.md / 13-step-9.md lack
  Step 6b fence parity'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3c1aa8ca3871
- workflow-fix
created_at: '2026-08-22T13:39:08Z'
has_clean_result: false
parent_id: 2263
origin_prompt: 'Surfaced by the #2263 round-4 code-review ensemble: Codex concern
  parent-reuse-fallback-parity (the 10-step-6.md ~:881 parent-reuse fallback omits
  repo-branch resolution, the launch recheck, and extra-sync threading, so verbatim
  use can refuse or under-stage) plus the Claude code-reviewer''s matching standing
  recommendation on two condensed launch references at 12-step-8.md:303 and 13-step-9.md:2889.
  Both rated non-blocking for #2263 and outside its reconciler-bounded scope.'
workflow: v1
---
## Overview / Motivation

Surfaced by the round-4 code-review ensemble on #2263 as a standing recommendation from the Claude `code-reviewer`.

**SCOPE NARROWED 2026-08-22 (before dispatch).** As originally filed, this task also covered the parent-reuse fallback block in `.claude/skills/issue/steps/10-step-6.md` (Codex concern `parent-reuse-fallback-parity`). That part was **pulled back into #2263 and fixed there** at round 6. Reason: `parent-reuse-fallback-parity` was a *raised CONCERN row* on #2263's `concerns.jsonl`, `task.py defer-concern` is USER-ONLY by spec, and an autonomous session may not complete a task with an open concern row it could resolve — and the site was in #2263's own deliverable file. Routing it out was the wrong disposition for the ledger half; this task keeps only the sites that genuinely live in OTHER files.

#2263 spent five rounds making three things mandatory at the primary Step 6b launch fence: the shared `--print-repo-branch` resolution, a mechanically-halting launch recheck, and `EXTRA_SYNC_ARGS` threaded into the dispatch argv. Two condensed launch references in other step files were never brought to that standard.

## Goal

Decide and apply the right disposition for each of the two condensed launch references outside `10-step-6.md`, so no copy-paste-able invocation in the workflow surface can dispatch without repo-branch resolution, the launch-fence recheck, and extra-sync threading.

## Workflow gap

Two sites, both pre-existing, neither introduced by #2263:

1. `.claude/skills/issue/steps/12-step-8.md:303`
2. `.claude/skills/issue/steps/13-step-9.md:2889`

Flagged together by the Claude `code-reviewer` at #2263 round 4 as a standing recommendation (non-blocking there, and outside that task's reconciler-bounded scope).

**Why it is a workflow gap.** These are operator-copyable command blocks in the workflow surface. #2263's central finding was that a gate certifying one input set while the dispatch consumes another is a hollow gate. A *condensed* launch example that silently drops the resolver or the sync threading reintroduces the same divergence channel at a second and third site — mechanically enforced at the primary fence, unenforced here.

## Scope caution for planning — do not assume a uniform sweep

Not every condensed reference *should* carry the full invocation. Some are plausibly legitimate abbreviations inside explanatory prose rather than copy-paste targets. Decide per site among:

- **(a) complete the invocation** — if it is genuinely a copy-paste launch target;
- **(b) replace with a pointer to the canonical Step 6b fence** — if it exists to illustrate an argument shape; one canonical site means nothing to drift;
- **(c) mark it explicitly illustrative-not-executable** — if it is prose scaffolding.

A mechanical sweep that inflates two prose references into two full fences would add surface without adding enforcement, and every duplicated fence is a new drift channel of the class #2263 fought for five rounds. #2263's round 6 made the same per-site judgment for the `10-step-6.md` block — **read what it decided and why before choosing here** (`epm:results v8` on #2263); consistency across the four sites matters more than any individual choice.

## Consider a mechanical pin

If the outcome is that N sites must carry the same tokens, a text-pin test over those sites is worth more than the edits. #2263's record is that prose parity claims decay: its round-2 fix put the lane suffix in a comment, and its round-3 prose falsely asserted the launched set "cannot drift". Check whether #2263 round 6 already added such a pin that this task should extend rather than duplicate.

## Coordination — probe before dispatching a writer

Open task **#2407** ("Step 6b canonical launch snippet is provision-only — add the required workload leg + time-budget guidance") targets the Step 6b launch snippet in `10-step-6.md`. Not a duplicate of this task (different gap, and after the narrowing above this task no longer touches `10-step-6.md` at all), but if planning here reaches for that file anyway, the two would collide as concurrent writers. Per `.claude/rules/cross-session-writer-arbitration.md`, run the pre-dispatch probe (`spawn_session.py list` + a `file-set claim:` marker scan + `git log --since` recency on the intended paths) and either sequence-after-commit or split to a disjoint file set.

## Verified at filing

- #2263 `events.jsonl`: `epm:code-review v4` (the standing recommendation naming both sites), `epm:code-review-codex v4` (concern `parent-reuse-fallback-parity`, now resolved on #2263 at round 6), `epm:results v8` (#2263 round 6's per-site disposition for the `10-step-6.md` block).
- `epm:review-reconcile v3` on #2263 — the binding adjudication that bounded round 4 to the primary fence, which is why these sites were left.

## Provenance

workflow_fix_target: .claude/skills/issue/steps/12-step-8.md

Routed from the #2263 round-4 review ensemble per `.claude/rules/workflow-fix-on-bug.md` — a surfaced-prose follow-up gets the same auto-file treatment as a formal candidate block; parking it as a chat note is the named anti-pattern. Non-blocking for #2263 by both reviewers' rating, and these two sites live outside #2263's deliverable file.
