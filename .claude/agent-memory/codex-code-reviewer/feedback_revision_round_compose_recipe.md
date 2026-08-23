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
- Claude-PASS / Codex-FAIL round with NO reconciler (r4 shape, #1739 a2fix
  r4 2026-08-22): the twin's OWN prior FAIL verdict is the SOLE work order
  envelope (`ROUND-3 CODEX VERDICT BODY`); the sibling PASS markers are
  named as context-never-evidence in the head. V1 replays the twin's OWN
  probe; a row the twin held at ADDRESSED-CLAIMED-BUT-UNVERIFIED solely on
  the fixed hole gets an explicit RE-ADJUDICATION instruction. Anchors =
  the prior round's compose-script NEW strings verbatim (14 sites, all
  guards green first run). Pre-adjudicate a MATCHING marker diff-stat too
  ("no discrepancy — do not manufacture one").
- FAIL+FAIL union round (no reconciler — r3 shape, #1739 a2fix r3
  2026-08-22): the work order is BOTH prior FAIL verdict bodies, each in its
  own envelope (`ROUND-2 CLAUDE VERDICT BODY` / `ROUND-2 CODEX VERDICT
  BODY`). Neutralize PROMPT-SIDE, not just by instruction: blockquote (`> `)
  every line-start `CONCERN:: ` row in BOTH inlined bodies, and replace the
  Codex body's own head/closing sentinel tags with bracketed notes — then
  assert `count(closing tag) == 1` on the final prompt. V-duty split: V1 per
  sibling blocker, V2 per own blocker (replay your OWN r2 mechanizable
  probes + bug-class sweep sites), V3 = no-over-tightening (new refusal
  gates must pass valid input — the fix-round calibration's other
  direction), V4 sweep+audit, V5 test substance.
- Implementer fail-pre-fix evidence citing a `/tmp` log (#1739 cms r3
  2026-08-22): Codex's worktree sandbox cannot read `/tmp`, so the composer
  verifies the log itself and INLINES the digest as a composer-verified fact
  — fail/pass counts + the verbatim failing-test ids + a rationale line for
  any legitimately-passing pre-fix test (coverage add vs defect pin) — and
  declares the `/tmp` path's unreachability explicitly NOT
  `data-access-blocked`. The twin's duty then shifts from existence
  ("did they fail") to CORRESPONDENCE ("does each fail against the base for
  the right gate", static `git show <base>:<path>` replay).
- TWO implementer markers in one round (fix commit + addendum commit):
  inline BOTH in the ONE standard `IMPLEMENTATION MARKER BODY` envelope with
  `=== IMPLEMENTATION MARKER k of 2 ===` separators (keeps the Step-3 grep
  guard intact); fail-pre-fix reads then take TWO base shas — the round base
  for commit 1's test claims, commit 1 for the addendum's.
- The prior round's patched rubric span extracts MECHANICALLY from its
  template: `assert tpl.startswith(head) and tpl.endswith(out)`, span =
  the middle — no saved span file needed.
- Nth-iteration SAME-bug-class round (#1739 a2fix r5, 2026-08-22 — 4th
  coverage-universe iteration): add an explicit CLASS-RESIDUAL SKEPTICISM
  head block (each prior fix closed the named probe and left a residual
  one level down ⇒ working prior: another residual exists — name WHERE to
  hunt: the set algebra itself, each gate's "the other gate owns it"
  boundary on BOTH sides, unregistered parents not just children, and
  consumers absent from the sweep table); when the impl marker carries a
  class-sweep table with why-safe dispositions, auditing EACH why-safe
  line in code becomes its own V-duty (a wrong why-safe is a blocker).
  Duty numbering follows the round's blockers, not the prior template's.
  Pre-adjudicate per-file diff-stat glosses that conflate
  insertions+deletions (v34's "+117/−31-ish" vs numstat +86/−31) when the
  TOTAL matches — presentational, never a discrepancy to manufacture.

Extends [[two-leg-single-label-round-compose]] (leg-suffixed filenames,
leg-scoped sentinel, round-matching by leg).
