---
title: 'workflow_lint: flag stale gotchas.md pointers after an entry relocates (live
  instance in research-project-structure.md)'
kind: infra
tags:
- wf-fix
- stale-relocation-pointer
created_at: '2026-08-08T02:54:51Z'
has_clean_result: false
origin_prompt: '#2189 round-1 code-reviewer prose follow-up: a lint flagging gotchas.md
  + #N co-references where #N no longer occurs in gotchas.md would mechanize the stale-relocation-pointer
  class (two live instances found this round).'
workflow: v1
---
# Goal

Mechanize the stale-relocation-pointer class: a lint flagging text that names
`gotchas.md` alongside a `#<N>` citation where `#<N>` no longer occurs in
`.claude/rules/gotchas.md`.

## Why now

#2189 relocated 27 entries verbatim out of `.claude/rules/gotchas.md` into six
topic-owning rule files (a 24.4 KB reduction), and its `## Out of scope`
section names **generalized residual-relocation** as a successor lever — so
more entries will move. Every relocation risks leaving behind a pointer that
still sends the reader to `gotchas.md` for a rule that now lives elsewhere.

Two live instances were found by hand during #2189's round-1 code review:

1. `.claude/rules/upload-policy.md:247` — pointed at "gotchas.md, #923" for the
   conditional-`.env` entry relocated to `.claude/rules/pod-side-reporting.md`.
   FIXED in #2189 commit `2ddec80aad` (it was in the round's own diff scope).
2. `.claude/rules/research-project-structure.md:73` — the same class
   (`` `.claude/rules/gotchas.md`; incident #923 ``), left UNFIXED because it
   was outside #2189's criterion-8 diff scope. **This is a live stale pointer
   on `main` right now** and is the natural first regression fixture.

Both were caught by a reviewer reading prose. Nothing mechanical detects them.

## A distinct, related instance worth handling in the same pass

`.claude/rules/LESSONS.md`'s `gotchas.md` row — the ALWAYS-ON index — still
advertises the topic "chained smoke-then-full leg out-root residue" among
gotchas.md's contents, though #2189 relocated that rule to
`.claude/rules/crash-fix-rounds.md`. #2189 could not fix it: editing
`LESSONS.md` was a hard prohibition in that task (43 B of headroom against
`_LESSONS_MAX_BYTES = 9600`, and further trimming of that row was ruled out at
#1269).

This is the same failure mode one level up — the index misdirects rather than a
sibling rule file — and it argues the check should cover LESSONS.md trigger
rows too, not just `#N` co-references. Note the LESSONS.md form has no `#N` to
key on, so it needs a different detector (topic-phrase presence), and fixing
any LESSONS.md row still runs into that file's byte budget. Consider whether
this half belongs here or with the LESSONS.md budget decision.

## Proposed shape (for the implementer to settle)

- Primary detector: for each `#<N>` that co-occurs with a `gotchas.md` mention
  in a tracked `.claude/**/*.md` file, FAIL (or WARN) when `#<N>` does not
  appear in `.claude/rules/gotchas.md`.
- Expect false positives: a pointer may legitimately cite `gotchas.md` for
  CONTEXT while the `#<N>` belongs to a different claim in the same sentence.
  Calibrate against the live tree before choosing FAIL vs WARN — a WARN that
  actually fires beats a FAIL that gets suppressed.
- Baseline is NOT clean: `research-project-structure.md:73` must either be
  fixed in the same change or explicitly grandfathered.

## Provenance

Filed by the #2189 orchestrator from the round-1 `code-reviewer`'s prose
follow-up (lower-confidence, uncapped per
`.claude/rules/workflow-fix-on-bug.md`). Sibling of #2192 (conflict-marker
residue lint), filed from the same review.
