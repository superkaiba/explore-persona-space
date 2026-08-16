---
name: claude-verifies-downgrade-safety-skips-071-enumeration
description: "Claude reviewers prove a dry_run/smoke gate-skip is mutation-SAFE but never hold the #2165 disclosure line — round-added branch unenumerated (r2) OR round-authored affirmative summary falsified by pre-existing branches (r3); verify against the full branch sweep (#2321 r2+r3)"
metadata:
  type: feedback
---

Claude code-review groups verify a smoke/dry-run conditional's SAFETY
(no mutation can leak) and stop there — nobody runs Step 0.71: is the
round-added downgrade NAMED in the module smoke blind-spot enumeration,
and is the enumeration's empty-form claim still true?

**Incident (#2321 r2):** the round added `if not dry_run:` around the C4
resume content-anchor probe (`issue2321_repack.py:1625-1640`; parent had a
bare `continue`). g2 proved the exemption structurally cannot reach a
deletion (true) and gave CONCERNS; g4 walked gates 0.5/0.55/0.6/0.8/0.9 —
not 0.71. The pre-existing enumeration (`:66-72`) still asserted "No
smoke-conditional branch substitutes an implementation or downgrades an
assertion" — now affirmatively false. Codex FAILed on exactly this
(`smoke-blind-spot-unenumerated`) and was upheld as blocking.

**Why:** `.claude/rules/smoke-blind-spots.md` (#2165, from #1336) pins this
class as a FAIL tag — substantive, never stripped — precisely because
"the downgrade is safe" is not a substitute for disclosure: a sanctioned
downgrade is still a blind spot, and the smoke's green PASS is cited as
launch evidence downstream.

**How to apply:** when reconciling a code-review split where Codex carries
`smoke-blind-spot-unenumerated` and Claude PASSes: (1) confirm the branch
is ROUND-ADDED — diff the parent commit's file (a pre-existing branch does
not trigger 0.71, though it can falsify the enumeration's empty-form
sentence); (2) read the enumeration block and check the specific skip is
named AND the "no downgrade" sentence is still true; (3) if unenumerated,
uphold as BLOCKING per the rule's fixed severity — do not downgrade to a
doc nit, even when the safety analysis is correct (the fix is one
docstring block + report mirror; cheap bounce, rule-pinned). Claude-side
mutation-safety findings and the Codex enumeration finding are COMPATIBLE,
not contradictory.

**Recurrence + refinement (#2321 r3):** the r3 fix round enumerated the
ordered items (g)/(h) but AUTHORED a new affirmative summary ("skips the
two Hub-probe gates named in (g)/(h)") that PRE-EXISTING dry-run branches
falsify — the pre-issue admission probe (`:1297` early-return before
`probe_unit_state` + drift/mixed/content-mismatch/window-drift aborts +
I18/I8 interlocks) and the #1739 liveness SystemExit/defer gate (`:2669`)
are also skipped. Claude g2 again ran the safety-not-disclosure argument
("raise LOUD on any live launch — disclosure not owed") — even after r2
was upheld against exactly that — and even LISTED the omitted sites in a
no-action minor. Upheld Codex FAIL again. Refinement to step (1): a
pre-existing branch does not trigger 0.71 by itself, but a ROUND-AUTHORED
affirmative claim in the enumeration (a count, "only", "the two …", or an
empty-form sentence) puts the module's WHOLE `dry_run`/`smoke` branch
inventory in scope — sweep all conditional sites and check the claim
against the full table before crediting closure. A Claude minor that
states the omitted facts "for completeness, no action" is the tell, not a
mitigation.
