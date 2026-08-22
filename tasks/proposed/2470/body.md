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

**SCOPE NARROWED 2026-08-22 (before dispatch).** As originally filed, this task also covered the parent-reuse fallback block in `.claude/skills/issue/steps/10-step-6.md` (Codex concern `parent-reuse-fallback-parity`). That part was **pulled back into #2263 and fixed there** at round 6. Reason: `parent-reuse-fallback-parity` was a *raised CONCERN row* on #2263's `concerns.jsonl`, `task.py defer-concern` is USER-ONLY by spec, and an autonomous session may not complete a task with an open concern row it could resolve — and the site was in #2263's own deliverable file. This task keeps only the sites that genuinely live in OTHER files.

#2263 spent seven rounds making three things mandatory at the primary Step 6b launch fence: the shared `--print-repo-branch` resolution, a mechanically-halting launch recheck, and `EXTRA_SYNC_ARGS` threaded into the dispatch argv. Two condensed launch references in other step files were never brought to that standard.

## READ FIRST — a live pin constrains your options

**#2263 added a UNIQUENESS invariant, and it will fail this task's most obvious fix.**

`tests/test_verify_carryover_inputs.py::test_step6_parent_reuse_fallback_points_at_canonical_launch_fence` asserts that **exactly ONE `dispatch_issue[.py] launch` INVOCATION** exists across ALL fenced blocks of the composed `/issue` spec, and that the block carrying it is the canonical Step 6b fence (resolver + `if !` recheck + `${EXTRA_SYNC_ARGS[@]+…}` all asserted).

The invariant is counted at **invocation** grain, not block grain, and it recognizes **both** the path spelling (`scripts/dispatch_issue.py launch`) and the module spelling (`python -m scripts.dispatch_issue launch`), across **bash / sh / console / bare** fences. Prose mentions outside fenced blocks are deliberately not counted.

Consequence: **completing either `12-step-8.md:303` or `13-step-9.md:2889` into an executable launch invocation WILL break that test — by any spelling, in any fence language.** #2263's round-5 review ensemble was asked whether that coupling is a feature or an accidental trap and ruled **feature**: one canonical fence is the accumulated lesson of #2263's rounds, and every duplicated fence is a fresh drift channel of the class those rounds kept closing.

So route (a) below is effectively closed unless you can argue the invariant itself should be relaxed — and that argument belongs in a plan reviewed against #2263's history, not in a quiet test edit. Do NOT widen or delete the pin to make room for a second invocation.

*(History note, for anyone reading an older revision of this body: at round 6 the pin counted only ` ```bash `-fenced BLOCKS, so a module-spelling or ` ```sh `-fenced completion would have escaped it, and an earlier revision of this section overstated its reach. #2263 round 7 closed all three escapes — second-invocation-in-same-block, module spelling, and non-bash fence — each verified to FAIL against the widened pin. The statement above is now accurate as written.)*

## Workflow gap

Two sites, both pre-existing, neither introduced by #2263:

1. `.claude/skills/issue/steps/12-step-8.md:303`
2. `.claude/skills/issue/steps/13-step-9.md:2889`

Flagged together by the Claude `code-reviewer` at #2263 round 4 as a standing recommendation (non-blocking there, and outside that task's reconciler-bounded scope).

A related residual named in #2263's `epm:results v8` (d), same surface, worth folding in: `13-step-9.md:2890` still carries an old intent-then-backend argv form in prose.

**Why it is a workflow gap.** These are operator-copyable command blocks in the workflow surface. #2263's central finding was that a gate certifying one input set while the dispatch consumes another is a hollow gate. A *condensed* launch example that silently drops the resolver or the sync threading reintroduces the same divergence channel at a second and third site — mechanically enforced at the primary fence, unenforced here.

## Scope caution for planning — do not assume a uniform sweep

Decide per site among:

- **(a) complete the invocation** — see the pin warning above; effectively closed.
- **(b) replace with a pointer to the canonical Step 6b fence** — what #2263 round 6 chose for the analogous parent-reuse block, after judging that block's role to be the reuse *decision* rather than a dispatch. One canonical site means nothing to drift.
- **(c) mark it explicitly illustrative-not-executable** — if it is prose scaffolding. #2263 round 6 paired its pointer with an `echo … >&2` line stating the block does not dispatch; reuse that shape if it fits.

Read #2263's `epm:results v8` for the reasoning behind its per-site choice. Consistency across all four sites matters more than any individual disposition.

Note that a **prose-only** mention is already outside the pin's reach, so disposition (c) needs no pin change — and if you convert a fenced block to prose, verify the pin still sees exactly one invocation afterwards.

## Consider extending, not duplicating, the pin

If your outcome is that N sites must carry the same tokens or the same pointer, **extend** the existing invariant rather than adding a parallel pin. #2263's record is that prose parity claims decay: its round-2 fix put the lane suffix in a comment, its round-3 prose falsely asserted the launched set "cannot drift", and its round-6 pin advertised an invariant stronger than it enforced. Any new assertion must be verified RED against the pre-fix text before it counts — a pin that cannot fail is the exact currency #2263 spent seven rounds learning not to accept.

## Coordination — probe before dispatching a writer

Open task **#2407** ("Step 6b canonical launch snippet is provision-only — add the required workload leg + time-budget guidance") targets the Step 6b launch snippet in `10-step-6.md`. Not a duplicate of this task (different gap, and after the narrowing above this task no longer touches `10-step-6.md`), but if planning here reaches for that file anyway, the two would collide as concurrent writers. Per `.claude/rules/cross-session-writer-arbitration.md`, run the pre-dispatch probe (`spawn_session.py list` + a `file-set claim:` marker scan + `git log --since` recency on the intended paths) and either sequence-after-commit or split to a disjoint file set. The same applies to #2263 itself if it is still in flight — it owns `10-step-6.md` and the pin test.

## Verified at filing

- #2263 `events.jsonl`: `epm:code-review v4` (the standing recommendation naming both sites), `epm:code-review-codex v4` (concern `parent-reuse-fallback-parity`, resolved at round 6), `epm:results v8` (round 6's per-site disposition + the `13-step-9.md:2890` residual), `epm:code-review v5` / `epm:code-review-codex v5` (the feature ruling on the coupling, plus the three pin-escape findings), `epm:results v9` (round 7's widened invocation-grain pin and its three escape-mutation verifications).
- `epm:review-reconcile v3` on #2263 — the binding adjudication that bounded round 4 to the primary fence, which is why these sites were left.

## Provenance

workflow_fix_target: .claude/skills/issue/steps/12-step-8.md

Routed from the #2263 round-4 review ensemble per `.claude/rules/workflow-fix-on-bug.md` — a surfaced-prose follow-up gets the same auto-file treatment as a formal candidate block; parking it as a chat note is the named anti-pattern. Non-blocking for #2263 by both reviewers' rating, and these sites live outside #2263's deliverable file.
