---
name: codex-fenced-content-blindness-on-plan-verbatim-validator
description: Codex FAILs a plan-verbatim raw-regex note/marker validator for fenced-code-block blindness; demote when the plan registered the raw-scan design and the arming audit ran the SAME fence-blind instrument over the full corpus with zero collisions (#2309 r1)
metadata:
  type: feedback
---

Codex flagged fenced-content blindness in a `re.MULTILINE` raw-regex marker
validator as BLOCKER (`markdown-structure-validator-bypass`, #2309 r1: a
fenced `### (d)` counts as present; a fenced `## Completion Report` quote
triggers applicability; a prose `part=1/3` disables the check). The behavior
VERIFIES by execution — but it was not a round defect: the approved plan's
Design section carried the exact regexes verbatim, pre-registered the
residual holes (`=`-only part token + measured prose-collision audit), and
armed a waiver flag as the escape for "any false positive".

**Why:** two independent demote grounds. (1) Plan-registered design — a
design-scope critique belongs at plan critique, not a code-review FAIL
(sibling: [[codex-methodology-choice-as-bug]], hardening-beyond-contract).
(2) Empirical bounding by instrument identity: the arming audit ran the SAME
fence-blind regex over the full historical corpus (~4,570 rows), so its
measured zeros (drain-signature 0 both kinds, part-token collisions 0)
bound the false-refusal population INCLUDING fenced shapes — the objection's
population is measured empty by the objection's own instrument. Also weigh
failure direction: a false pass degraded to the status quo (the mechanical
check ADDS to human review), a false refusal cost one waiver retry.

**How to apply:** when Codex FAILs a validator/gate diff for a parsing
blind spot, check (a) is the regex/predicate in the plan's Design section
verbatim? (b) did the plan's arming/calibration audit run the same
instrument over the population the blind spot would hit, and measure zero?
(c) is each failure direction fail-open or escape-valved? All three → demote
to Standing-only hardening. Companion instance same round: a wrong argv in
MARKER PROSE ("--json" vs the committed CLI's --tasks-root/--out) upheld as
non-blocking when the script's docstring is correct and the counts reproduce
independently — the [[codex-conflates-marker-format-with-code]] class; also
Codex fabricated a sibling claim that the plan's Reproducibility Card
carried the wrong command (it carried no audit command at all — verify
sibling-claim citations against the plan text before crediting them).
