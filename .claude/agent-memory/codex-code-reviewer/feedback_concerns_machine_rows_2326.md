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

**DELIMITED-block form (#2646; first ordered on #2387 r11, 2026-08-30):**
`persist_verdict_concerns.py` can silently no-op at rc=0 ("persisted 0/0")
when rows fail its recognition — #2646 (filed, blocked) is the fail-loud
fix. Until it lands, a brief may order "the delimited concerns block form";
no repo-canonical delimiter grammar exists, so the COMPOSER defines it in
the template: keep the `## Concerns to persist` heading (the forwarder's
`--require-block` keys on heading+rows), wrap the rows in two exact
delimiter lines `CONCERNS-BLOCK-BEGIN` / `CONCERNS-BLOCK-END` (plain text,
never an HTML comment — the tag-extraction grep must not see extra `<!--`
lines), instruct rows ONLY between them + `CONCERN:: none` as the sole-row
empty form, assert each delimiter ==1 in the composed prompt, and REPORT
the delimiter tokens to the orchestrator in the return so extraction keys
on them. A brief's "or write ledger rows directly via task.py" alternative
NEVER applies to the read-only twin — override it in the
Codex-adaptations-of-the-brief block.
