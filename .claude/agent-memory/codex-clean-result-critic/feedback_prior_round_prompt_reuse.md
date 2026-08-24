---
name: prior-round-prompt-reuse
description: Rounds 2+ — reuse the previous round's prompt file (/tmp/codex-prompt-issue-<N>-crc<r>.md) as the compose base, with per-lens live-file currency assertions replacing a full rebuild.
metadata:
  type: feedback
---

For round r >= 2 of the same task, compose from the round-(r-1) prompt
file (it persists at `/tmp/codex-prompt-issue-<N>-crc<r-1>.md`) via span
replacement instead of rebuilding from the live sources — the pattern
the code-reviewer twin runs (#2476 r2→r4) and this composer ran at
#2476 crc4.

**Why:** the base already carries the Lens 13/14 compose patches and
the full inline set; span replacement touches only the round-specific
blocks (history, scope, dismissals, envelopes, sentinel), so patch
regressions and splice bugs fail loud in `rep1`/`replace_span`
count==1 asserts. The "never compose from a frozen lens list" rule is
discharged MECHANICALLY, not by trust: assert live lens sections 1-12
+ 15 are verbatim substrings of the base (`section.rstrip() in P`),
assert the Lens 13/14 patched pointers present, and assert live
SPEC.md verbatim-contained — any upstream drift since the prior round
crashes the compose instead of shipping stale rubrics. Also confirm
`git log -1` + clean `git status` on the three sources vs the base
prompt's mtime before trusting containment.

**How to apply:** #2476 recipes at `/tmp/codex-2476-crc{4,5}-compose.py`
(crc5 = latest; ran clean 2026-08-23 r5, incl. a fresh dismissal row +
envelope execution-error assert).
Round-specific spans to replace: PRIOR CRITIQUE SUMMARIES block (ends
at `\n\nAll paths above`), ROUND-N SCOPE + BINDING DISMISSALS block
(ends at the `=== INLINED` banner), the three envelopes (BEGIN..END
inclusive, fresh Step 1d captures), the marker sentinel + Round
heading, and the "re-runs on rounds N-10" window. Finish with a
stale-string sweep (assert old round tokens absent). Note: the Step 4
global `{{` scan legitimately hits ~6 lines of verbatim SPEC content
(the spec's own no-`{{`-sentinel rules) — only the envelope-scoped
placeholder check is binding. Related: [[lens13-plan-fetch-patch]],
[[Delta-scoped rounds beyond r3 — compose, don't hard-fail]].
---
name: prior-round-prompt-reuse
description: Rounds 2+ — reuse the previous round's prompt file (/tmp/codex-prompt-issue-<N>-crc<r>.md) as the compose base, with per-lens live-file currency assertions replacing a full rebuild.
metadata:
  type: feedback
---

For round r >= 2 of the same task, compose from the round-(r-1) prompt
file (it persists at `/tmp/codex-prompt-issue-<N>-crc<r-1>.md`) via span
replacement instead of rebuilding from the live sources — the pattern
the code-reviewer twin runs (#2476 r2→r4) and this composer ran at
#2476 crc4.

**Why:** the base already carries the Lens 13/14 compose patches and
the full inline set; span replacement touches only the round-specific
blocks (history, scope, dismissals, envelopes, sentinel), so patch
regressions and splice bugs fail loud in `rep1`/`replace_span`
count==1 asserts. The "never compose from a frozen lens list" rule is
discharged MECHANICALLY, not by trust: assert live lens sections 1-12
+ 15 are verbatim substrings of the base (`section.rstrip() in P`),
assert the Lens 13/14 patched pointers present, and assert live
SPEC.md verbatim-contained — any upstream drift since the prior round
crashes the compose instead of shipping stale rubrics. Also confirm
`git log -1` + clean `git status` on the three sources vs the base
prompt's mtime before trusting containment.

**How to apply:** #2476 recipes at `/tmp/codex-2476-crc{4,5}-compose.py`
(crc5 = latest; ran clean 2026-08-23 r5, incl. a fresh dismissal row +
envelope execution-error assert). Dead-twin rounds (#823 r6,
2026-08-24): a round whose Codex job died AFTER compose still leaves a
valid prompt base — reuse it for round r+1, and add an explicit
"INDEPENDENCE GAP — NOW CLOSED" REVIEW CONTEXT block: no fabricated
verdict exists, this is the first Codex review of the cycle, verify the
surviving Claude verdict's findings as inputs rather than defer to
them, and state the head-sentinel-vs-posted-version offset (head = brief
round; posted top-level = auto max+1 on the codex kind, which trails
when a round never posted).
Retraction rounds (#823 r9, 2026-08-24): the reuse BASE is the prompt AS
DISPATCHED, so it can carry orchestrator-injected blocks the composer
never wrote (r8 shipped an `=== ORCHESTRATOR ADDENDUM ===` with a
"use it; do not redo it" verification block) — when the brief retracts a
fence or an attestation, EXCISE the inherited block entirely, move any
still-true facts (the 13-pin roster, the HF-revision-vs-git-SHA probe
caveat) into the REVIEW CONTEXT as neutral facts, and re-probe them
fresh rather than copying the attestation. And when the retraction
narrative deliberately QUOTES the retracted instruction, the
stale-string sweep cannot ban the phrase outright — scope it: assert
count == 1 AND the negating sentence ("No such instruction exists this
round") is present.
Round-specific spans to replace: PRIOR CRITIQUE SUMMARIES block (ends
at `\n\nAll paths above`), ROUND-N SCOPE + BINDING DISMISSALS block
(ends at the `=== INLINED` banner), the three envelopes (BEGIN..END
inclusive, fresh Step 1d captures), the marker sentinel + Round
heading, and the "re-runs on rounds N-10" window. REVIEW-CONTEXT
splice gotcha (#823 r7): the block's `(round N — ...)` header line can
WRAP onto a second line, so never assert `lines[h+1] == sep` — locate
the closing `=` sep within the next ~3 lines instead. Finish with a
stale-string sweep (assert old round tokens absent). Note: the Step 4
global `{{` scan legitimately hits ~6 lines of verbatim SPEC content
(the spec's own no-`{{`-sentinel rules) — only the envelope-scoped
placeholder check is binding. Related: [[lens13-plan-fetch-patch]],
[[Delta-scoped rounds beyond r3 — compose, don't hard-fail]].
