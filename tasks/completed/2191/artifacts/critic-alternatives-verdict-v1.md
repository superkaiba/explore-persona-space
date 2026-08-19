# Critic verdict — Alternatives + robustness lens (plan v1)

Agent: `critic` (Claude), lens = Alternatives / false-FAIL surface / simpler shapes.
Codex twin: CONFIRMED NO-SHOW (quota sentinel live at compose time; Claude-only
decision per the site's no-show fallback).

**Rating: REVISE** (2 Must-Fix, both one-line plan edits; 1 nice-to-have removal)

## Must-Fix 1 — check (b1) needs a mode split (promote-time false-FAIL channel)

Check (b1)'s FAIL compares a FROZEN report against an EXTERNAL MUTABLE reference
set. Any post-generation follow-up round or inline analysis that writes one more
full-40-hex non-dirty reproducibility card under `eval_results/issue_<N>/**`
makes a previously-PASSing, UNCHANGED report FAIL at `--mode promote` — the task
body's named worst outcome (blocking promotion of a good result).

The plan's own §2 cites `_check_pin_blob_identity` as "the model for a git-object-DB
check with a mode-split degrade ladder", but §4.2's signature
`check_code_sha_cards(raw_body, blanked_lines, figures_root, expect_issue)` and
the §4.3 wiring snippet thread no `mode`.

Fix: thread `mode` into check (b). b1 is FAIL at `generation`, `is_warn=True` at
`promote` (b2/b3 already WARN).

Check (a) needs NO split — its FAIL (resolvable pin + empty ls-tree) matches the
model check's "pinned commit does not contain path" arm, which the house pattern
deliberately keeps FAIL in both modes.

Mechanizable: build a tmp repo with one uncited usable card; assert
`check_code_sha_cards(..., mode="promote").is_warn is True` and `mode="generation"`
FAILs.

## Must-Fix 2 — check (a)'s negation guard is too narrow

§4.1 guards only `(?<!not )(?<!never )`, and the §4.1 not-fire enumeration
over-claims "Negated claims" as covered. Unguarded, all matching the trigger:

- "nothing is committed under `X`"
- "isn't / wasn't in git under `X`"
- "rather than / instead of in git under `X`"
- "never landed in git under `X`" (intervening word defeats the immediate lookbehind)
- "no longer committed under `X`"

These FAIL whenever the line also carries a resolvable pin and the path is empty —
and that co-occurrence is REALISTIC, not hypothetical: the check's own FAIL message
instructs authors to reword false git-home claims into HF-home sentences, which
naturally keep the empty git path for contrast. Worked example that would FAIL on a
CORRECT, convention-compliant sentence:

    scores live on HF under `X` rather than in git under `.../scores/` at `20fcef9c28...`

Fix (inside the plan's pre-authorized "shrink the fire surface" deviation space):
replace the two lookbehinds with a preceding-window negation scan (~30-40 chars
before the match; token set `not|n't|never|no longer|rather than|instead of|nothing`),
and add one criterion-5 fixture row pinning a "rather than in git under
`<empty path>` at `<pin>`" line as NON-firing. Over-broad windows only under-fire —
the sanctioned direction.

## Nice-to-have (not blocking) — drop b3

b3 is machinery the incidents do not justify: the plan itself proves b3 stays
SILENT on the incident's own defective segment (no token of "grid"/"anchors"
resolves to `gates/pilot_gate_report.json`), it has zero confirmed catches, and its
stopword set is `ungrounded — needs smoke-test` by the plan's own §11. b1 (coverage
FAIL) + b2 (row-scope WARN) catch both the incident and the observed mid-correction
state. If b3 IS retained, its stopword set must be pinned by criterion-5 tests, not
left "tunable at implementation".

## Must-answer #1 — the (b) false-FAIL channel, case by case

- Launcher / aborted-leg / gate-re-run / upload-only card, commit irrelevant to
  reported numbers: FAIL under the plan (if full-40-hex, non-dirty). Partially
  correct: gate reports CANNOT be excluded (the incident's own card IS
  `gates/pilot_gate_report.json`), and upload cards carry the workload commit
  (reporting-relevant in #2162). The genuinely-false residue — a halted/superseded
  attempt's card — is mostly closed by overwrite-in-place card paths and crash-fix
  stale-artifact wipe discipline. In the observed #2162 population the only
  "irrelevant" cards are 8-hex/dirty, already excluded. Acceptable at generation
  once Must-Fix 1 removes the promote channel.
- Run spanning many commits, report summarizes: FAIL until the per-phase split
  enumerates every usable card commit. CORRECT — the task body itself endorses this
  ("the expected output is a per-phase split"); intended norm, not a false FAIL.
- Sibling issue's tooling writing under this issue's tree: FAIL — incorrect, but
  low-likelihood (lint-guarded, `--check-upload-prefix-clobber`). Concern-level.
- Refs stale / absent / ahead: absent -> PASS-note N/A (correct; designed under-fire
  on sparse clones). Stale-behind -> working-tree union covers. Ahead-of-report ->
  the promote-staleness false FAIL, i.e. Must-Fix 1.

## Must-answer #2 — the (a) channel and the HF carve-out

The trigger splits the mixed line 37 CORRECTLY: the git clause matches; the HF
clause's preceding bigram is "repo under", which the regex structurally cannot
match — the HF path can NOT be mis-attributed as a claimed git home. Designed
under-fires (all acceptable per the body): "tracked in git at", "committed at/to",
"in the repo under", "in-git `file`", no-backtick paths, ellipsis-abbreviated paths,
slashless bare filenames, hyphenated "in-git under". Wrongly-fires: the negation
family in Must-Fix 2. Gitignored-vs-false-claim IS distinguishable: a force-added
gitignored file appears in ls-tree (passes); an absent one makes the claim genuinely
false (correct FAIL). Committed-on-an-invisible-branch: the pin must be on the
claim's own line; an unresolvable pin -> WARN, never FAIL.

## Must-answer #3 — is there a simpler shape?

Check (a): every piece of machinery is incident-grounded (brace expansion, branch
tokens, URL-hex stripping each trace to a real line in the one existing report);
any-pin-satisfies only loosens. KEEP. Check (b): drop b3 (above). The
narrower-assert alternative (hardcode the two confirmed instances) is REJECTED —
it would be dead on the next report; the §4.2 coverage redesign is measured, not
stylistic, because the body's literal row-to-card pairing is provably unresolvable
for the incident's own phase naming. The extensibility claim is supported:
one-function-per-check + one dispatch line is the file's demonstrated idiom across
20 existing checks.

## Also assessed

- verify_plan WARN (unscoped pytest green): v1 criterion 7 / §5 row 4 already read
  a path-scoped invocation over the single test file. Add one sentence stating the
  plan-time green baseline of that file so "fully green" is baseline-safe.
- Weakening existing checks: NONE. The §4.3 wiring is append-only between the live
  dispatch lines; `body` is in scope for the (b) call; Assumption 6 (`_by_name`
  lookups, not positional) plus the criterion-7 warn-audit covers the residual.
- Grandfathering is moot: #2162 is the first-ever v2 report and it is the
  acceptance SILENT fixture; no report predates the checks.

## Independent re-verification the critic performed

- `git ls-tree -r 20fcef9c28 -- eval_results/issue_2162/judge/` = 82 blobs;
  `.../judge/scores/` = 0 blobs; `origin/issue-2162` resolves (`db5d1680a2...`).
- The live report's line 37 and Code-SHAs row match the plan's reading, EXCEPT the
  row has mutated again since the plan's Assumption-3 quote — the mutation the
  plan's risk row anticipates; the freeze-at-implementation rule handles it.

## Concerns for the implementer (NOT blocking)

1. "Fires on the incident" is fires-on-a-faithful-RECONSTRUCTION for both FIRE
   fixtures (round-1 draft unrecoverable; Assumption 5 honestly Medium on that
   side). The honesty note in criterion 1 must survive into the test docstrings
   verbatim.
2. Check (a)'s accepted residue: a SUBSET claim over a non-empty directory
   ("scores committed under `judge/`") is structurally invisible to a
   path-emptiness check — so the check may not have caught the round-1 sentence if
   it named only `judge/`. Accepted per the body's falsifiability definition; carry
   as a scope note in the implementer report.
3. Sibling-issue cards under this issue's tree would false-FAIL b1; worth one line
   in the FAIL message's "if this phase is covered elsewhere" clause.
4. Criterion 8's CLI test resolves `figures_root` to the real repo root via
   `_default_figures_root` when the frozen fixture lives inside the repo — either
   pass an explicit tmp `--figures-root` or accept the real-object-DB dependency
   consciously.
5. If b3 is retained, pin its stopword set by criterion-5 tests.

## Orchestrator addendum (independently verified after the verdict landed)

- Must-Fix 1's precision CONFIRMED at `scripts/verify_report.py:645-680`: the
  mode-split arms are exactly as described, and the `does not contain {path}` arm
  (line 656) carries NO mode branch — FAIL in both modes. So the asymmetric
  prescription (split (b), do not split (a)) matches the file's own idiom rather
  than imposing a new one.
- Wiring availability CONFIRMED: `mode` is a keyword param of `verify_report_text`
  (line 930) and in scope at the append point, alongside `body`, `blanked_lines`,
  `blanked_body`, `figures_root`, `expect_issue`. `check_manifest`'s
  `if manifest_path is not None` guard (line 971) is the absent-artifact degrade
  model the plan already cites.
- The mutation claim CONFIRMED and STRENGTHENED: the Code-SHAs row now reads
  "analysis outputs at consolidation commit `b228639eac...` (ancestor of the branch
  pin `20fcef9c28...`, `issue-2162`)" vs the plan's quoted "at branch HEAD". File
  grew 48,570 -> 52,826 bytes, mtime 2026-08-07 19:51 PT, still untracked (`??`).
  The #2162 session is live-editing the ONLY realistic input, so
  freeze-at-implementation is load-bearing, not hygiene. The new row is also a
  BETTER stress case: two SHAs on one line, one ellipsis-abbreviated.
