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

**SCOPE NARROWED 2026-08-22 (before dispatch).** As originally filed, this task also covered the parent-reuse fallback block in `.claude/skills/issue/steps/10-step-6.md` (Codex concern `parent-reuse-fallback-parity`). That part was **pulled back into #2263 and fixed there** at round 6, because it was a raised CONCERN row on #2263's own ledger, `task.py defer-concern` is USER-ONLY, and the site was in #2263's own deliverable file. This task keeps only the sites that live in OTHER files.

#2263 spent seven rounds making three things mandatory at the primary Step 6b launch fence: the shared `--print-repo-branch` resolution, a mechanically-halting launch recheck, and `EXTRA_SYNC_ARGS` threaded into the dispatch argv. Two condensed launch references in other step files were never brought to that standard.

## READ FIRST — the one-canonical-fence rule is a CONVENTION here, not a guardrail

**Corrected 2026-08-22 (third revision of this section — read the history note below before trusting any earlier revision).**

#2263 added `tests/test_verify_carryover_inputs.py::test_step6_parent_reuse_fallback_points_at_canonical_launch_fence`, whose docstring advertises "exactly ONE `dispatch_issue[.py] launch` invocation across ALL fenced code blocks" of the composed `/issue` spec.

**What it actually enforces, as verified independently by both #2263 round-6 reviewers:** occurrences of a launch-shaped regex inside **column-zero, exactly-three-backtick** fences with a bare single-word info string. It recognizes the path spelling and the module spelling.

**What it does NOT catch — and this is the part that matters for this task:** an **INDENTED** fenced block. Also four-backtick fences, tilde fences, extended info strings, a line-continued invocation, and quoted program/subcommand forms. And it over-fires in the other direction: a *commented* launch-shaped line inside a recognized fence turns it RED despite adding no invocation, so it counts regex occurrences rather than shell invocations.

**Both of this task's two anchor sites are indented list content.** Codex verified by mutation that inserting a valid indented Bash fence at each exact anchor left the test GREEN. The composed spec currently carries 184 indented fence delimiter lines.

**Therefore: completing either site into a normally-indented fenced example would create a second operator-copyable launch site that the pin will NOT flag.** The one-canonical-fence rule still holds as the accumulated lesson of #2263's rounds — its round-5 ensemble ruled the coupling a feature, and every duplicated fence is a fresh drift channel of the class those rounds kept closing — but you must honor it **deliberately**. Do not plan on the test stopping you.

Practical consequence for your options below: prefer (b) or (c). If you nonetheless choose (a), you owe an explicit argument against #2263's history AND a pin that actually catches your new block — and #2263 may itself widen the detector (see below), so check its final state first.

### History note — two earlier revisions of this section overstated the pin, and one may have been read

Revision v2 said the pin asserts "exactly ONE **bash block** carries `dispatch_issue.py launch`" and that completing either reference into a fence WILL break it. False then: the round-6 pin counted `bash`-fenced BLOCKS and the path spelling only, so a module-spelling or ` ```sh ` completion escaped.

Revision v3 said the round-7 pin made that true "by any spelling, in any fence language." Also false: round 7 widened spelling and fence *language* but stayed column-zero-anchored, so the indented case — which is exactly this task's case — still escapes.

Both revisions handed this task a mechanical guarantee that did not exist. Recorded rather than silently re-tightened, because a planner who read either revision needs to know which guarantee they were relying on. **#2263 review round 6 is where both overstatements were caught** (Claude `r7-uniqueness-pin-residual-escapes`, Codex `r6-uniqueness-pin-2470-coupling` re-raised as `verified-open`); whether #2263 widens the detector to cover indented fences is being adjudicated there. Re-read `epm:results` and the round-6/7 verdicts on #2263 before relying on any statement in this section.

## Workflow gap

Two sites, both pre-existing, neither introduced by #2263:

1. `.claude/skills/issue/steps/12-step-8.md:303`
2. `.claude/skills/issue/steps/13-step-9.md:2889`

Flagged together by the Claude `code-reviewer` at #2263 round 4 as a standing recommendation (non-blocking there, outside that task's reconciler-bounded scope). Neither site is fenced today — both are indented prose.

A related residual named in #2263's `epm:results v8` (d), same surface, worth folding in: `13-step-9.md:2890` still carries an old intent-then-backend argv form in prose.

**Why it is a workflow gap.** These are operator-copyable command blocks in the workflow surface. #2263's central finding was that a gate certifying one input set while the dispatch consumes another is a hollow gate. A *condensed* launch example that silently drops the resolver or the sync threading reintroduces the same divergence channel at a second and third site.

## Scope caution for planning — do not assume a uniform sweep

Decide per site among:

- **(a) complete the invocation** — see the READ FIRST warning: permitted only with an argued case plus a pin that genuinely catches the new block.
- **(b) replace with a pointer to the canonical Step 6b fence** — what #2263 round 6 chose for the analogous parent-reuse block, after judging that block's role to be the reuse *decision* rather than a dispatch.
- **(c) mark it explicitly illustrative-not-executable** — if it is prose scaffolding. #2263 round 6 paired its pointer with an `echo … >&2` line stating the block does not dispatch; reuse that shape if it fits.

Prose-only mentions are outside the pin's reach entirely (13 such mentions exist today, deliberately unpinned), so (c) needs no pin change. Read #2263's `epm:results v8` for the reasoning behind its per-site choice; consistency across all four sites matters more than any individual disposition.

## Consider extending, not duplicating, the pin

If your outcome is that N sites must carry the same tokens or pointer, **extend** the existing invariant rather than adding a parallel pin. #2263's record is that these claims decay: round 2 put the lane suffix in a comment; round 3's prose falsely asserted the launched set "cannot drift"; round 6's pin advertised block-grain coverage it did not have; round 7's pin advertised "ALL fenced code blocks" while staying column-zero-anchored. Any new assertion must be verified RED against the pre-fix text before it counts, **and the mutation must be the one that matters** — for this task that means an INDENTED fence at the real anchor site, not an unindented specimen.

## Coordination — probe before dispatching a writer

Open **#2407** ("Step 6b canonical launch snippet is provision-only") targets the Step 6b snippet in `10-step-6.md`. Not a duplicate (and after the narrowing this task no longer touches that file), but if planning reaches for it anyway the two collide as concurrent writers. Per `.claude/rules/cross-session-writer-arbitration.md`, run the pre-dispatch probe (`spawn_session.py list` + a `file-set claim:` marker scan + `git log --since` recency on the intended paths) and either sequence-after-commit or split. The same applies to #2263 while it is in flight — it owns `10-step-6.md` and the pin test.

## Verified at filing

- #2263 `events.jsonl`: `epm:code-review v4` (the standing recommendation naming both sites), `epm:code-review-codex v4` (`parent-reuse-fallback-parity`, resolved at round 6), `epm:results v8` (round 6's per-site disposition + the `13-step-9.md:2890` residual), `epm:results v9` (round 7's widened pin), `epm:code-review v6` + `epm:code-review-codex v6` (the indented-fence escape, demonstrated by mutation at this task's exact anchor sites).

## Provenance

workflow_fix_target: .claude/skills/issue/steps/12-step-8.md

Routed from the #2263 round-4 review ensemble per `.claude/rules/workflow-fix-on-bug.md`. Non-blocking for #2263 by both reviewers' rating, and these sites live outside #2263's deliverable file.
