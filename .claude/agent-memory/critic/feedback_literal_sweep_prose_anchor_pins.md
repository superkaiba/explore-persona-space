---
name: literal-sweep-prose-anchor-pins
description: Cap-raise / literal-sweep plans — test pins anchor on PROSE literals (.index("Max 5 rounds per reviewer"), _region(..., "(rounds 2-5)")) invisible to token greps; grep tests/ for EVERY edited literal family and replay the plan's own verification grep for its expected residual (#2391 v2)
metadata:
  type: feedback
---

For any plan that rewrites a numeric/token literal across many surfaces (a
review-round cap raise, a trigger rename, a marker-string sweep), TWO
independent replays are mandatory before APPROVE:

1. **Prose-anchor pin hunt.** The plan's coupled-test grep is typically
   token-keyed (`grep -rn 'cap-5\|cap5\|cap_5' tests/`) — but skill-text pin
   tests anchor REGION SLICES on prose literals the sweep rewrites:
   `text.index("Max 5 rounds per reviewer", start)` raises ValueError the
   moment the literal becomes "Max 10 rounds"; a `_region(text, start,
   "**If REVISE (rounds 2-5):**")` helper fail-louds on `end != -1`. #2391
   v2 missed BOTH (`tests/test_issue_skill_neutral_gate_vocab_brief.py:93`,
   `tests/test_issue_skill_humanize_verify_first_pin.py:63`) despite a §7
   "grep-verified" claim — the token grep structurally cannot see prose
   anchors. Check: grep tests/ for EVERY edited "Current fragment" family
   (`max 5 rounds|rounds 2-5|round cap 5|up to 5 rounds|count[<>]=?5|at
   round 5|after round 5|cap \(5\)`), and map every hit to
   edit-in-same-commit or an out-of-scope class. #784's equivalent claim
   ("no test pins == 3") was true THEN; the pin population grows, so the
   claim must be re-derived each raise, never inherited.

2. **Expected-residual replay of the plan's own verification grep.** When
   an acceptance criterion keys on "sweep returns ONLY <enumerated lines>"
   (0 unexplained hits), run the plan's grep VERBATIM on the live tree and
   map every hit: #2391 v2's §8 cmd 2 scanned src/scripts/tests with bare
   `cap-5` (matches `cap-512`) and `cap \(5\)` (matches the infra
   dispatch-loop cap's live string `file_infra_task.py` + 2 test pins + a
   watcher-test comment), plus the re-keyed cap test's own pattern strings
   — ~10 legitimate residual hits absent from the enumeration, making AC
   "0 unexplained hits" unsatisfiable as registered. Sibling of checklist
   items M (sweep-output tabulation, #2156) and I2 (self-FAILed insertion,
   #917/#2285) in [[infra-plan-review-checklist]].

**Why:** both defects are invisible to reading the plan and cheap to catch
by replay (2× 2-min greps); the failure lands as post-hoc suite red or an
implementer round burned re-classifying — or worse, "fixing" out-of-scope
look-alikes.

**How to apply:** any plan whose §-table rewrites >~10 literal sites with a
coupled-test claim or an expected-residual enumeration; run both replays
before verdict.

**Round-3 addenda (#2391 v4):** (1) round-3 on a converged literal-sweep plan
is PURE REPLAY — extract each authored command block VERBATIM from the plan
file (`sed -n 'A,Bp' plan.md > f; bash f pre|post`), never retype; a plan
whose §16 records rc + outputs per repaired command replays in ~10 tool
calls. (2) Selector-COVERAGE attributions need a per-file probe:
`select_step9c_tests.py`'s `WORKFLOW_SURFACE_GLOBS` (incl.
`.claude/workflow.yaml`) is a SKIP/short-circuit list, NOT a mapping arm — a
workflow.yaml-only `--map-files` probe returns zero mappings, so "covered by
the workflow-surface mapping" is not a real channel; real channels are
`WORKFLOW_INVARIANT` registry membership + the skills/rules-pin discovery
arms. (3) A residual-sweep's pattern family is never total over doc-prose
edit sites ("(cap 5)", "round-5 cap", "max 5 per the", "On round 5 (the
cap)") — hunk-presence review against the full §-table is the gate for
wholly-missed sites; only partial-application-inside-a-hunk and wrapped-form
regression need named §4.2-style focus items.

**Round-2 addenda (#2391 v3 — three revision-introduced instrument traps):**
(1) a `\b` appended to a token pattern (`cap_5\b`) silently un-matches
`cap_5_surface` — `_` is a word char; replay the plan's EXACT pattern
strings, never a "tidied" variant. (2) a bare digit-less alternate
(`|loops up to`) in a residual-enumerated grep matches the FIXED site
post-edit → off-by-one vs a registered "EXACTLY N lines" AC; simulate the
post-edit state of every alternate. (3) baseline-subtracted lint compares
keyed on `grep -E '^(FAIL|WARN)'` are FAIL-blind: `workflow_lint.py` error
records print as `workflow_lint: <err>` + terminal
`workflow_lint: FAIL (N error(s))`; only `WARN: ...` lines match — capture
`^workflow_lint: ` minus `: PASS$` instead. Also check pin-grep FILE paths
(a cmd-3 look-alike grep aimed at issue-tick/SKILL.md was vacuous — the
string lives in steps/13-step-9.md), and check that a post-edit pair-scan
(cmd 2b) excludes the re-keyed test file whose negative-control FIXTURES
reproduce the wrapped stale forms.
