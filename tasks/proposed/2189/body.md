---
title: 'Decide the gotchas.md byte budget: a real re-trim pass or a deliberate cap
  raise'
kind: infra
tags:
- gotchas-byte-budget
created_at: '2026-08-08T00:51:27Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from task #2185''s implementer: gotchas.md
  headroom is 318 B after #2185''s forced in-task re-trim, so any next append (including
  an automatic failure-lesson promotion) re-trips test_live_tree_passes_clean fleet-wide
  and forces the next unrelated task into an unreviewed editorial pass.'
workflow: v1
---
# Decide the gotchas.md byte budget: a real re-trim pass or a deliberate cap raise

## Goal

`.claude/rules/gotchas.md` is operationally out of headroom. Decide the budget
policy once, in a reviewed change, so the next unrelated task that touches the
file is not forced into an emergency editorial pass.

## The state

- Budget: WARN above `GOTCHAS_SIZE_WARN_BYTES = 200_000`, FAIL above
  `GOTCHAS_SIZE_FAIL_BYTES = 250_000` (`scripts/workflow_lint.py`, both
  strictly-greater, bundled into the no-flags run).
- `tests/test_workflow_lint_gotchas_size.py::test_live_tree_passes_clean` pins
  the LIVE tree to **zero WARNs**, so crossing 200,000 B fails a committed test
  that the Step 9c mapped-test gate selects.
- Headroom before #2185: **340 B**. After #2185's forced in-task re-trim:
  **318 B**.

## Why this is a workflow gap, not just a full file

The file is machine-appended by `scripts/consolidate_lessons.py` (failure-lesson
promotion), so it regrows without any human deciding to grow it. At 318 B of
headroom, essentially ANY next append — including a single automatic
failure-lesson promotion — re-trips the pinned test fleet-wide. Whichever task
happens to touch `gotchas.md` next then inherits an unreviewed trim of unrelated
safety lessons as a side effect of its own unrelated work.

That is exactly what happened on **#2185**: a ~2.2 KB trap entry (the whole
deliverable) did not fit, so that session had to condense its own bullet AND
compress 15 other entries by ~3.4 KB to land at all. Those cuts were
archaeology-only, pointer-backed and pin-grepped, and they follow the check's own
printed editorial policy — but they were an editorial pass on a shared safety
surface driven by byte pressure rather than by a decision to make them.

## The two options

1. **A real re-trim pass** targeting ~10-15 KB of headroom, executed per the
   check's own recipe: keep the operative rule + diagnostic signature + fix +
   bare `#N` citations; drop dates, session ids, wall-times, and fix-status
   archaeology (resolve to current state); collapse superseded/FIXED entries to
   one line. The file still carries substantial incident narrative whose
   long-form lives in `.claude/agent-memory/**` — verify the pointer exists
   before each cut, as #2185 did.
2. **A deliberate cap raise**, with `test_threshold_literals_pinned` updated in
   the same change and the rationale recorded. Note the double lock is
   intentional: the literals are pinned by a test precisely so that raising the
   budget cannot happen absentmindedly. The check's docstring frames itself as
   "the backstop that forces a periodic re-trim", so a raise is a policy change
   and should read as one.

Option 1 is the presumption — it is what the instrument asks for. Option 2 is
defensible only with an argument that the ledger's growth is legitimate and that
200,000 B was arbitrary; if taken, raise it enough to be a decision rather than
another 300 B of runway.

## Scope notes

- Do NOT loosen the FAIL threshold as a shortcut, and do NOT delete an operative
  rule, diagnostic signature, fix, or `#N` citation to buy bytes.
- Before each cut, confirm the removed long-form actually exists somewhere
  durable (the named agent-memory file), and grep the removed literals against
  `tests/` + `scripts/workflow_lint.py` so no pin is silently lost.
- `.claude/rules/LESSONS.md` is a separate, tighter problem (43 B of headroom
  against `_LESSONS_MAX_BYTES = 9600`, with the `gotchas` index row at exactly
  its 1,175 B grandfather cap, and a further trim of that row ruled out at
  #1269). Do not fold it into this task; note it if this task's decision
  changes the picture.

## Provenance

Filed by the #2185 orchestrator from that session's implementer
`workflow-fix-candidate` block. #2185 landed its deliverable and its own forced
trim; this task owns the durable budget decision that #2185 should not have had
to make.
