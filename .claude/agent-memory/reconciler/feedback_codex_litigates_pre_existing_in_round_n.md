---
name: codex-litigates-pre-existing-in-round-n
description: Codex code-reviewer correctly identifies a real silent-failure / fail-fast / hygiene violation, but escalates it to a round-N blocker when the violating code is unchanged from trunk and only adjacent to round-N's actual diff
metadata:
  type: feedback
---

Codex sometimes correctly identifies a real silent-failure pattern (warn-and-skip-instead-of-raise, swallowed exception, missing assert), correctly cites the relevant CLAUDE.md rule (e.g. "fail fast — never hide failures"), and correctly judges that the pattern is harmful. But it FAILs round-N specifically over the pattern even when the violating block is **unchanged from trunk** and only **adjacent to** the actual round-N diff.

**Why:** Codex's reviewer prompt asks "does this diff violate project conventions?" and it answers the broader question "does this FILE violate conventions?" — picking up pre-existing patterns that happen to be touched, viewed, or referenced by the diff. The reviewer's per-round context lacks a clean diff/trunk separation; it sees the file as a whole and judges the whole.

**How to apply:** When Codex flags a real fail-fast / silent-failure / hygiene violation as a round-N blocker:
1. Run `git show main:<path> | sed -n '<lines>p'` (or the relevant base ref) and check whether the EXACT block Codex is flagging is present in trunk.
2. If yes — pre-existing — the finding is at most a Real-but-non-blocking observation. Surface as a standing recommendation, not a blocker. The right venue is a separate cleanup task scoped to the contract, not a force-revert of a bug fix.
3. If no — newly introduced or materially worsened in round N — Codex is correct and FAIL is the right verdict.
4. Check if round N **marginally** broadens the pre-existing pattern (e.g. round 3 of #385 added isinstance gates that route a narrow class of mixed-type rows from a loud crash to the pre-existing warn-skip). If the marginal regression is data the system doesn't actually produce, classify as Real-non-blocking. If real production data hits the new silent path, escalate.
5. Concrete incident: task #385 round 3, Codex FAILed on warn-and-skip-instead-of-raise in `format_dataset`. The warn-skip else branch (`trainer.py:269-274`) was unchanged from trunk and has been shipping since at least `f16e8d47` (codebase refactor). Round 3 only added a new TRL conversational `elif` branch (fixing the actual smoke crash) plus isinstance gates. The marginal new silent-skip path (mixed-type prompt/completion rows) is not produced by any current generator. PASS, with standing recommendation to open a separate cleanup task converting warn-skip to raise + adding the missing `pytest.raises(ValueError)` test.

Distinguish from [[feedback_codex_scope_drift_on_repeat_findings]] (Codex mis-attributes which prior round's fix scope applies, lexical-match driven) — this pattern is broader: Codex litigates a long-standing project-wide pattern using the round-N diff as the venue. Distinguish from [[feedback_claude_underclasses_silent_failures]] (Claude under-flags NEW silent failures): that calibration applies when the silent failure is new in the diff; the pattern here is the opposite — Codex over-escalates when the silent failure is pre-existing.
