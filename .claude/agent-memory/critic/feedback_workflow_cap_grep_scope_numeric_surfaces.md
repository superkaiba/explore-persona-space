---
name: Workflow cap-rename grep-scope misses numeric-comparison surfaces
description: A workflow-fix "rename cap N" plan whose acceptance grep matches only prose ("cap 3", "N rounds") silently misses the machine-read numeric surfaces (round<3, revision_round>=3, count<3, entry_condition strings) — a #622 sibling-hit class
type: feedback
---

When reviewing a workflow-fix / kind:infra plan that RENUMBERS a round cap
(or any policy constant) across the workflow surface, the acceptance set is
insufficient if its verification grep + tests match only the PROSE phrasings
of the constant ("cap 3 per reviewer", "Round cap 3", "Max 3 rounds") and
NOT the machine-read numeric-comparison surfaces that encode the SAME
behavior with a bare integer:

- `revision_round < N` / `revision_round >= N` (SKILL.md loop guards)
- `round < N` / `round >= N` (resume-semantics tables)
- `count < N` / `count >= N` (step-graph FAIL branches)
- `entry_condition: "... revision_round < N"` (workflow.yaml step-graph
  string conditions — these are driver-consumed, not just doc prose)
- resume-table rows encoding the OLD cap-hit TERMINAL (e.g.
  `revision_round>=3 | error path | failure-exit`) that the new behavior
  replaces — left unchanged they contradict the new terminal.

**Why it's conclusion-changing for an infra plan:** a stale `< 3` left in
one of these files silently PASSes the acceptance grep + the prose-only
"no stale cap 3" test, so a broken change (two configs disagreeing — prose
says continue to round 5, the driver-read entry_condition still stops at 3)
ships as correct. This is the #622 sibling-hit class (a literal-pattern
constant living in N surfaces; the grep named only some) applied to the
ACCEPTANCE set rather than the edit set.

**The check to demand (mechanizable):** the acceptance grep pattern AND the
regression test must include the numeric-comparison alternations, scoped to
the in-scope loop:
`grep -rnE 'revision_round\s*[<>]=?\s*3|round\s*[<>]=?\s*3|count\s*[<>]=?\s*3|cap 3|N rounds' <surface>`
and a `test_no_stale_cap_3_numeric` reading the edited files' regions and
asserting no `< 3` / `>= 3` survives in the renamed loop. The plan may
enumerate the numeric surfaces in its edit set (§3) yet STILL leave the
acceptance grep prose-only — enumeration in the edit set does not substitute
for a matching regression check, because a manual edit can miss one.

**How to apply:** run the numeric-alternation grep YOURSELF against the
workflow surface (`.claude/ CLAUDE.md scripts/ src/`) during review;
cross-check every hit against the plan's §3 edit enumeration. Any numeric
surface in the same loop that the plan does NOT enumerate is a Must-Fix
(stale-constant-ships-silently). Origin: #784 (cap 3→5) — the §7 acceptance
grep + §8 test #3 matched only prose; independent grep found
`workflow.yaml:2568 entry_condition "code-review FAIL, revision_round < 3"`
and `SKILL.md:7119 revision_round>=3 → failure-exit` both un-enumerated and
uncaught by the acceptance set.
