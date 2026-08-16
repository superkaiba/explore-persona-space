---
name: claude-verifies-downgrade-safety-skips-071-enumeration
description: "Claude reviewers prove a round-added dry_run/smoke gate-skip is mutation-SAFE but never run Step 0.71's enumeration check — the #2165 smoke-blind-spot FAIL class; verify the branch is round-added (parent diff) and the enumeration text names it (#2321 r2)"
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
