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

**How to apply:** #2476 recipes at `/tmp/codex-2476-crc{4,5,6,7,8}-compose.py`
(crc8 = latest; ran clean 2026-08-24 r8 = round 1 of a SECOND fold re-gate,
k200 census. New r8 patterns: (1) a FOURTH envelope — `COMPOSER ARTIFACT
DIGEST` — appended after OPEN-CONCERNS carrying JSON headline fields + a
numpy digest + derived reads for branch-only round artifacts (npz included;
use np.nanmedian for corpus-half arrays — plain median NaNs and reads as a
false alarm); patch the "three envelopes" intro to "four" and extend the
Step-4 semantic guard loop to the 4th name. (2) Worktree eval JSONs
attested BYTE-IDENTICAL to the result-commit pins (git hash-object per
file) become a sanctioned Codex read path — say so explicitly, since the
#922 rule otherwise bans working-tree reads. (3) HF-only cited digits
(train_log.json dead fractions) fetched + attested at compose time.
(4) A brief-specified verdict vocabulary (`needs_targeted_fix — <n>
fixes`) is grafted as a bracketed instruction AFTER the standard bold
Verdict line, never replacing the spec template. crc7 = FIX-VERIFICATION
round shape: scope block = fix roster (each fix -> its r6 blocker, with
realizing anchors + an explicit adjudicate-on-the-merits note where the
analyzer took a path DIFFERENT from the r6 edit text) + delta-confinement
attestation (diff prior-reviewed body commit -> current, hunk count +
regions, /tmp mirror extracted WITH frontmatter) + compose-time scrub greps
attested (0-occurrence checks for banned phrases). A REGENERATED figure
gets a fresh /tmp extraction dir (r7-figs) beside the unchanged r6-figs
set, all blobs RE-verified against their pins at this round's compose.
Stale-sweep caveat (r8): sweep tokens must not substring-match the NEW
history block's own legitimate mentions of the prior fold's name.)

**Status-folder path staleness (crc6, 2026-08-24):** the reuse base
embeds the task's ABSOLUTE body/plan paths in THREE places — the header
BODY/PLAN lines AND the Lens 13 patched plan-path pointer (which quotes
the header path verbatim mid-lens). A task that changed status folders
between rounds (`interpreting` → `followups_running` when a fold round
started) leaves all three stale; re-derive via `task.py find` and rep1
each, and keep `tasks/<old-status>/<N>` in the stale-string sweep so a
missed occurrence crashes the compose (that sweep is what caught the
Lens 13 one). The old interpretation /tmp filename is a fourth same-class
slot.
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
