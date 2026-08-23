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

**How to apply:** #2476 crc4 recipe at `/tmp/codex-2476-crc4-compose.py`.
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
