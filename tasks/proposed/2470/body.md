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

## READ FIRST — a live pin covers the ordinary fix, with two named gaps

**Revised 2026-08-22, fourth revision of this section. The three earlier revisions were each wrong, in both directions — read the history note below before trusting anything you may have read here previously.**

The pin is `tests/test_verify_carryover_inputs.py::test_step6_parent_reuse_fallback_points_at_canonical_launch_fence`: exactly ONE `dispatch_issue[.py] launch` invocation may exist across the composed `/issue` spec's fenced blocks, and it must be the canonical Step 6b fence.

**What it enforces, after #2263 round 8** — verified by mutation, independently, by both round-7 reviewers and the round-7 reconciler:

- Fenced blocks at **any indentation**, backtick **or** tilde, any info string, unclosed-to-EOF handled.
- Counted as **invocations**, not textual hits: comments are stripped and line continuations joined before counting.
- Both the path spelling (`scripts/dispatch_issue.py launch`) and the module spelling (`python -m scripts.dispatch_issue launch`).

**For this task specifically:** an indented Bash fence inserted at **each** of the two anchor sites below was verified RED under the new pin and GREEN under the old one. Your anchors sit at 2 spaces (`12-step-8.md`) and 5 spaces (`13-step-9.md`, continuation inside a `- ` item), and both are covered. So if you complete either reference into an ordinary fenced example, **the pin will stop you** — this is a real guardrail for the ordinary case, not a convention.

**Two gaps remain, and one of them is adjacent to your work:**

1. **List-marker-prefixed fences and blockquoted fences still count 0** (`r8-uniqueness-pin-remaining-syntax-escapes` on #2263 — adjudicated a standing NIT, deliberately not closed, zero live instances). A fence written on its own indented line under a list item **is** caught; a fence written inline after the list marker (`- ```bash`) is **not**. Since `13-step-9.md:2887-2891` is list content, that distinction is one keystroke away from your edit. Write the fence on its own line.
2. Variable-assembled invocations count 0. Not a plausible shape for a doc example, listed for completeness.

**Two caveats on the guarantee itself:**

- **It attaches to the MERGED state.** `main` still carries the pre-#2263 pin until #2263 completes Step 10d. Check which version is live before relying on it: `grep -n "_fenced_blocks" tests/test_verify_carryover_inputs.py` on the tree you are actually working in.
- Prose mentions outside fences are deliberately uncounted (13 exist today). Both of your anchors are currently prose whose backtick span wraps across lines, which is why they count 0 now.

**Practical consequence for your options below.** Route (a) is no longer effectively closed, but it now trips a real pin, so taking it means arguing the invariant should be relaxed — that argument belongs in a reviewed plan, not a quiet test edit. Do NOT widen or delete the pin to make room for a second invocation. (b) and (c) remain the cheaper paths.

### History note — three earlier revisions of this section were wrong, and any of them may have been read

- **v2** claimed the pin asserts one **bash block** and that any fence completion breaks it. False then: the round-6 pin counted ` ```bash `-fenced BLOCKS and the path spelling only, so a module-spelling or ` ```sh ` completion escaped.
- **v3** claimed round 7 made that true "by any spelling, in any fence language." Also false: round 7 widened spelling and fence language but stayed **column-zero-anchored**, so the indented case — this task's actual case — still escaped.
- **v4** correctly described that column-zero state and concluded the rule here was "a CONVENTION, not a guardrail." Accurate when written, then falsified in the opposite direction about two hours later: #2263 round 8 widened the detector to any-indent fences, which is exactly what v4 said did not exist.

Three revisions, two directions of error, one root cause: each described a mechanism whose reach had not been adversarially verified. That is the same defect #2263 exists to fix, and the reason this section now cites per-claim mutation evidence and names its gaps rather than asserting a clean guarantee. Ledger trail on #2263: `r6-uniqueness-pin-2470-coupling` (raised r5, wrongly closed r7, reopened `verified-open` r6, upheld BLOCKER by reconciler v6) and `r7-uniqueness-pin-residual-escapes`.

If #2263 is still open when you plan this, re-read its latest `epm:results` and reconciler verdict before relying on any statement above.

## Workflow gap

Two sites, both pre-existing, neither introduced by #2263:

1. `.claude/skills/issue/steps/12-step-8.md:303`
2. `.claude/skills/issue/steps/13-step-9.md:2889`

Flagged together by the Claude `code-reviewer` at #2263 round 4 as a standing recommendation (non-blocking there, outside that task's reconciler-bounded scope). Neither site is fenced today — both are indented prose.

A related residual named in #2263's `epm:results v8` (d), same surface, worth folding in: `13-step-9.md:2890` still carries an old intent-then-backend argv form in prose.

**Why it is a workflow gap.** These are operator-copyable command blocks in the workflow surface. #2263's central finding was that a gate certifying one input set while the dispatch consumes another is a hollow gate. A *condensed* launch example that silently drops the resolver or the sync threading reintroduces the same divergence channel at a second and third site.

## Scope caution for planning — do not assume a uniform sweep

Decide per site among:

- **(a) complete the invocation** — see READ FIRST: this now trips a live pin for the ordinary indented-fence case, so it is permitted only with an argued case for relaxing the one-canonical-fence invariant, made in a reviewed plan.
- **(b) replace with a pointer to the canonical Step 6b fence** — what #2263 round 6 chose for the analogous parent-reuse block, after judging that block's role to be the reuse *decision* rather than a dispatch.
- **(c) mark it explicitly illustrative-not-executable** — if it is prose scaffolding. #2263 round 6 paired its pointer with an `echo … >&2` line stating the block does not dispatch; reuse that shape if it fits.

Prose-only mentions are outside the pin's reach entirely (13 such mentions exist today, deliberately unpinned), so (c) needs no pin change. Read #2263's `epm:results v8` for the reasoning behind its per-site choice; consistency across all four sites matters more than any individual disposition.

## Consider extending, not duplicating, the pin

If your outcome is that N sites must carry the same tokens or pointer, **extend** the existing invariant rather than adding a parallel pin. #2263's record is that these claims decay: round 2 put the lane suffix in a comment; round 3's prose falsely asserted the launched set "cannot drift"; round 6's pin advertised block-grain coverage it did not have; round 7's advertised "ALL fenced code blocks" while staying column-zero-anchored; round 8's widened the detector correctly but then described its comment handling as matching bash, which a `;#` comment falsifies. Any new assertion must be verified RED against the pre-fix text before it counts, **and the mutation must be the one that matters** — for this task that means an indented fence at the real anchor site, not an unindented specimen.

The transferable lesson #2263's round-7 reconciler drew, having watched seven rounds of this: **claims that state their exact rule and disclose their regex survive adversarial review; claims that paraphrase a regex into readable prose fail.** If you write a new pin here, describe what it matches, not what it means.

## Coordination — probe before dispatching a writer

Open **#2407** ("Step 6b canonical launch snippet is provision-only") targets the Step 6b snippet in `10-step-6.md`. Not a duplicate (and after the narrowing this task no longer touches that file), but if planning reaches for it anyway the two collide as concurrent writers. Per `.claude/rules/cross-session-writer-arbitration.md`, run the pre-dispatch probe (`spawn_session.py list` + a `file-set claim:` marker scan + `git log --since` recency on the intended paths) and either sequence-after-commit or split. The same applies to #2263 while it is in flight — it owns `10-step-6.md` and the pin test.

## Verified at filing

- #2263 `events.jsonl`: `epm:code-review v4` (the standing recommendation naming both sites), `epm:code-review-codex v4` (`parent-reuse-fallback-parity`, resolved at round 6), `epm:results v8` (round 6's per-site disposition + the `13-step-9.md:2890` residual), `epm:results v9` (round 7's widened pin), `epm:code-review v6` + `epm:code-review-codex v6` (the indented-fence escape, demonstrated by mutation at this task's exact anchor sites).

## Provenance

workflow_fix_target: .claude/skills/issue/steps/12-step-8.md

Routed from the #2263 round-4 review ensemble per `.claude/rules/workflow-fix-on-bug.md`. Non-blocking for #2263 by both reviewers' rating, and these sites live outside #2263's deliverable file.
