---
title: 'workflow-fix: parent-reuse fallback + condensed launch refs lack Step 6b fence
  parity (repo-branch, recheck, extra-sync)'
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

Surfaced by the round-4 code-review ensemble on #2263 (Codex concern `parent-reuse-fallback-parity`, plus a matching standing recommendation from the Claude `code-reviewer`). Both reviewers rated it non-blocking for #2263 and it is outside that task's reconciler-bounded scope — #2263's round-4 implementer had independently flagged the same site as out of bounds — so it routes here rather than widening a narrow fix round.

#2263 hardened the Step 6a.5 carryover gate and the Step 6b launch fence so that the gate, the launch recheck, and the dispatch all grade ONE consistent input set: a shared `--print-repo-branch` resolver, a mechanically-halting recheck, and `EXTRA_SYNC_ARGS` threaded into the launch argv. Other launch invocations in the skill were never brought to that same standard.

## Goal

Bring every remaining launch / dispatch invocation in the `/issue` skill steps up to the parity standard #2263 established for the Step 6b fence, so that no copy-paste-able invocation in the workflow surface can dispatch without the repo-branch resolution, the launch-fence recheck, and the extra-sync threading.

## Workflow gap

Three known sites, all pre-existing and none introduced by #2263:

1. **`.claude/skills/issue/steps/10-step-6.md` — the parent-reuse fallback example (~`:881`).** Per Codex: it omits repo-branch resolution, the launch recheck, and extra-sync threading. Used verbatim, it can either refuse (no resolved `REPO_BRANCH`) or under-stage (an rsync lane materializing without the plan-named `--extra-sync-path` values), which is exactly the class #2263 closed at the primary fence.
2. **`.claude/skills/issue/steps/12-step-8.md:303`** and **3. `.claude/skills/issue/steps/13-step-9.md:2889`** — two condensed launch references flagged by the Claude `code-reviewer` as a standing recommendation in the same round.

**Why it is a workflow gap.** These are operator-copyable command blocks in the workflow surface. #2263's whole finding was that a gate certifying one input set while the dispatch consumes another is a hollow gate — and a *condensed* launch example that silently drops the resolver or the sync threading reintroduces the same divergence channel at a second site. The primary fence is now mechanically enforced; these are not.

**Scope caution for planning.** Not every condensed reference necessarily *should* carry the full invocation — some may be legitimate abbreviations in explanatory prose rather than copy-paste targets. Part of this task is deciding, per site, whether the right fix is (a) complete the invocation, (b) replace it with a pointer to the canonical Step 6b fence, or (c) mark it explicitly as illustrative-not-executable. Do not assume (a) uniformly; a mechanical sweep that inflates three prose references into three full fences would add surface without adding enforcement.

**Consider a mechanical pin.** #2263's lesson is that prose parity claims decay — its round-2 fix put the lane suffix in a comment and its round-3 prose falsely claimed the launched set "cannot drift". If the outcome here is that N sites must carry the same tokens, a text-pin test over those sites is worth more than the edits themselves.

## Verified at filing

- #2263 `events.jsonl`: `epm:code-review-codex v4` (concern row `parent-reuse-fallback-parity`, persisted to `concerns.jsonl`), `epm:code-review v4` (the standing recommendation naming `12-step-8.md:303` and `13-step-9.md:2889`), and `epm:results v6` (the round-4 implementer recording `:881` as outside the reconciler's bounds).
- `epm:review-reconcile v3` on #2263 — the binding adjudication that bounded round 4 to the primary fence, which is why these sites were deliberately left.

## Provenance

workflow_fix_target: .claude/skills/issue/steps/10-step-6.md

Routed from the #2263 round-4 review ensemble per `.claude/rules/workflow-fix-on-bug.md` — a surfaced-prose follow-up gets the same auto-file treatment as a formal candidate block, and parking it as a chat note is the named anti-pattern. Non-blocking for #2263 by both reviewers' own rating; filed as a distinct task because the affected surface spans three step files beyond #2263's deliverable.
