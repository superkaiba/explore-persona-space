---
name: revision-round-compose-recipe
description: Round 2+ fix-round compose — reuse the prior round's /tmp template via assert-guarded string deltas; round-scope the diff to the fix commit(s); inline the binding reconciliation as the work order; claimed-addressed concern rows get an explicit verification-duty block
metadata:
  type: feedback
---

For review round 2+ on the same leg/label, do NOT re-derive the prompt from
`code-reviewer.md` — reuse the prior round's saved template
(`/tmp/codex-code-reviewer-<N>-<leg>-r<k>-template.md`, rubric span intact)
and apply ROUND-DELTAS via a Python compose script whose every replacement
is `assert span.count(old) == 1`-guarded, with a post-patch stale-reference
guard (`assert bad not in span` for the prior round's marker version, base
SHA + probe forms, duty-roster name, and "round 1 of a fresh leg" phrasing).
Validated on #1739 a2fix r2 (2026-08-22): 15 replacements, zero rubric
re-derivation, ~10 min compose. Re-validated on #1739 cms r2 (2026-08-22):
22 replacements (span + output section patched separately — anchor-count
asserts run per part, since `## Issues Found`-style anchors repeat across
parts), all guards green first run.

**Why:** the rubric span is ~68 KB of verbatim-copied text — re-deriving it
risks the #606 twin-omission class, while unguarded `sed`-style edits
silently miss (the r1 template itself carried leftover inconsistencies from
an earlier compose). Assert-guarded deltas fail loud instead.

**How to apply:**
- Diff scope = the FIX range only (`<r1-head>..<fix-head>`); BAN
  whole-branch / `origin/main...HEAD` bodies (they re-include the prior
  round's already-reviewed delta). The prior round's commits move to the
  Step 0.9 git-provenance base; update all three probe SHAs.
- Inline the binding reconciliation (`epm:code-review-reconcile` body) as
  its own envelope — the work order; instruct Codex to verify against IT +
  the plan, never the implementer's restatement.
- Replace the duty roster (L*) with fix-round duties (V*): one per
  sustained blocker (VERIFIED-FIXED, replaying the reconciler's concrete
  failure cases statically against the NEW code + fail-pre-fix reads via
  `git show <base>:<path>`), one per marker-(d) judgment call (adjudicate,
  named blocker-if condition), a regression sweep over prior-round-verified
  surfaces, a fail-open audit of every NEW branch, and a claimed-addressed
  ledger block.
- Claimed-addressed concern rows: extend the Step 0.8 status vocabulary
  with ADDRESSED-VERIFIED / ADDRESSED-CLAIMED-BUT-UNVERIFIED (substantive
  finding); marker "ADDRESSED" prose without a ledger `addressed` event is
  a composer-observed bookkeeping note (Minor max), never marker-shape.
- Pre-adjudicate mechanical mismatches you observe at compose time (head
  sentinel digit vs posted top-level version — #1739 r2 had v27-in-body vs
  posted 28; marker diff-stat vs composer-measured range) as
  at-most-CONCERNS lines in the facts section, so the twin cannot
  manufacture a marker-shape FAIL from them.
- Sentinel increments leg-scoped (`v2` for the leg's round 2); Step 4.5
  flips to "binds with full force" (BLOCKER-fix round); Step 4(b)/(c)
  marker-claims text updated to the new marker's (c).
- When inlining the twin's OWN prior verdict as settled context (a
  reconciler-demotion round), STRIP its `<!-- epm:code-review-codex v<k> -->`
  / closing tags from the inlined copy (a v1 tag in the prompt trips
  sentinel-count validation and can confuse extraction) and instruct: never
  copy its historical `CONCERN:: ` rows — the line-start token in the OUTPUT
  is reserved for the new round's persist section.
- Pin surrounding-code reads to the RANGE HEAD (`git show <head-sha>:<path>`)
  and word the zero-post-range-commits fact as "at compose time": the
  composer's own same-turn agent-memory commit (and any sibling bookkeeping)
  lands ABOVE the range before Codex reads, so a live-HEAD claim goes stale
  between compose and dispatch.

Extends [[two-leg-single-label-round-compose]] (leg-suffixed filenames,
leg-scoped sentinel, round-matching by leg).
