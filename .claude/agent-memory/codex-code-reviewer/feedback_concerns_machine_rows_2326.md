---
name: concerns-machine-rows-2326
description: Since #2326 (2026-08-16 19:13 PT) the verdict template MUST carry the CONCERN:: machine-row grammar in ## Concerns to persist plus a **Prior-concerns ledger:** header field — rubric-currency-check the composer spec itself, not only code-reviewer.md
metadata:
  type: feedback
---

Commit `2454922e7d` (#2326) changed BOTH review-side surfaces mid-fleet:

1. `codex-code-reviewer.md` output schema: `## Concerns to persist` is now
   machine-parsed — LINE-START rows
   `CONCERN:: <BLOCKER|CONCERN|NIT> <kebab-id> <summary <=180c>` (token 1 =
   severity, token 2 = kebab id, remainder = summary); zero concerns = the
   exact literal sole row `CONCERN:: none`; the token must never start a
   line outside that section (the orchestrator's
   `scripts/persist_verdict_concerns.py` blind-forwards `^CONCERN:: ` rows
   from anywhere in the extracted marker block). Prose bullets alone are NO
   LONGER persisted.
2. `code-reviewer.md` Step 7 header: new REQUIRED
   `**Prior-concerns ledger:** <K open: ids>` / `empty` field (the Step 0.8
   walk record) — add it to the composed template's header fields.

**Why:** on #2321, 8 emitted concerns reached nobody (0 persisted) because
nothing parsed the prose section; #2333 r1's 13 concerns likewise never
reached concerns.jsonl — the r2 compose had to fall back to the
ledger-empty closure pattern with the r1 verdict body as the acceptance
contract. Hit live on #2333 r2 (2026-08-16): the loaded composer spec
predated the change; only the recipe's rubric-currency check
(`git log -1 -- .claude/agents/code-reviewer.md` vs the prior compose's
mtime) surfaced it.

**How to apply:** at every compose, run the currency check over BOTH
`.claude/agents/code-reviewer.md` AND `.claude/agents/codex-code-reviewer.md`
(the composer's own spec — a template-side contract can change without the
rubric changing) and fold any post-extraction template deltas into the
fresh tail. Validate `CONCERN:: ` rows by CONTENT, not a fixed count: assert
the grammar row (`^CONCERN:: <BLOCKER|CONCERN|NIT>`) is present INSIDE the
template's `## Concerns to persist` section, plus zero line-start rows in
the plan/marker envelopes — the line-start count varies with template
wording (1 on #2514 r1, where `CONCERN:: none` sat mid-line in backticked
prose; 4 on the #2333-era template), so a pinned count false-FAILs a valid
compose. Related: [[revision-round compose recipe]],
[[bypath-brief-frozen-events-resolution]].

**#2523 addendum (2026-08-24, #2327 r1→r2):** the r1 composed prompt handed
Codex a KEY-VALUE row template (`CONCERN:: concern_id=… severity=… …`) while
the same prompt's later prose said the forwarder position-parses —
`persist_verdict_concerns.py` rejected both rows (`bad-severity` + `bad-id`
+ `duplicate-id`) and the ledger was hand-written. The positional grammar at
`codex-code-reviewer.md:625` is authoritative over ANY caller brief. Guard
shape that works (r2): for every line-start `CONCERN:: ` row in the
assembled prompt, assert tokens 1 AND 2 contain no `=`; a blunt
`'concern_id=' not in prompt` substring ban false-FAILs on the tail's own
prohibition prose ("no `concern_id=`, no `severity=` …"), which is worth
keeping — ban the ROW shape, not the substring.
