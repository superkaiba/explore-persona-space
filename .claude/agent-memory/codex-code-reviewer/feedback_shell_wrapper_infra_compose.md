---
name: shell-wrapper-infra-compose
description: Compose adaptations for kind:infra diffs touching a cron/shell wrapper (.sh) — Step 0.70 binds, live-alert + crontab-mutation never-run bans, seam=child-binary 3.8 shape, class-sweep completeness settle, scanner FN/FP axes
metadata:
  type: feedback
---

Compose adaptations for a `kind: infra` round whose diff touches a
cron/shell wrapper `.sh` (first used #2196 r1; extends
[[infra-wf-fix-lint-gate-compose]], which covers the workflow_lint.py
shape):

1. **Step 0.70 (smoke-variable gating) BINDS** — its trigger is "any `.sh`
   in the diff". Inline it verbatim even when the N/A-by-type block marks
   the experiment-only gates off; do not let it ride the N/A block.
2. **Live-alert never-run warning:** when the wrapper has a push/alert side
   channel (telegram_push.sh, PushNotification), the SAFETY block names the
   script explicitly — "never run `bash scripts/<wrapper>.sh` or the
   marker's repro commands; the default push path fires a LIVE alert" — on
   top of the generic never-execute-smoke-commands instruction. The
   marker's own repro line often carries a `/bin/true` pin precisely
   because the unpinned form alerts the user.
3. **Step 3.8 mapping for shell-wrapper test harnesses:** the healthy seam
   is the CHILD BINARY (`EPS_..._BIN=/bin/true`-style env stubs), with the
   REAL wrapper body executed. Instruct Codex to verify (a) the tests
   invoke the real `.sh`, and (b) env-var NAMES the tests set match the
   names the script reads (a mismatched env name makes a test green while
   pinning nothing).
4. **Shell-specific Step 2 scrutiny list:** quoting/word-splitting of
   expanded vars in new helpers, `set -e/-u` interaction with `|| fatal`
   arms, ordering claims (a probe that CREATES a file can flip
   first-run-of-day detection), best-effort push not masking exit codes,
   and exit-code propagation vs the preserved arms (compare against
   `git show origin/main:<path>`).

**Why:** the generic template is python-centric; #2196 r1's load-bearing
surface was all bash semantics, and the repro command in the marker would
have fired a live Telegram alert if the reviewer ran it unpinned.
**How to apply:** any compose where the round diff touches `scripts/*.sh`
(cron wrappers, dispatch shells, guard hooks).

**#2387 r1 (2026-08-28) sharpenings — CLASS-SWEEP variant (the round bounds
one call class across N wrappers; #2387 wrapped 10 `telegram_push.sh`
execution sites in `timeout --kill-after=5s "${PUSH_TIMEOUT}s"` across 6
cron wrappers + behavioral/scanner tests + selector registration):**

5. **The execution ban needs a SECOND hazard class beyond the live alert:
   crontab mutation.** Self-retiring watch wrappers
   (`cron_watch_issue_<N>.sh`) end their terminal arms with
   `crontab -l | grep -v ... | crontab -` — running one deletes live
   monitors from the user's real crontab. Name BOTH hazards, plus "never
   run `crontab` in any form", plus the `$HOME` log/sentinel writes. Item 2's
   live-alert wording alone does not obviously cover a reviewer who runs the
   wrapper "just to see the retire path".
6. **A class-sweep round gets a composer COMPLETENESS SETTLE, not a
   completeness question.** Re-run the class-boundary grep yourself
   (`git grep -l <helper> -- scripts/`), classify every hit, and attest which
   callers are already bounded — INCLUDING ones the plan never named. #2387's
   plan named 5 already-bounded Python callers; the composer found 2 more
   (`pod_audit.py:369`, `runpod_api.py:189`, both `timeout=20`) plus one
   comment-only `.sh` hit. Handing that as a settled fact closes a
   false-incompleteness blocker channel on the one claim the task exists to
   make, while explicitly inviting a finding if the twin finds a caller the
   composer missed.
7. **Attest the actual `set` line — never let the twin assume `set -e`.**
   All six #2387 wrappers run `set -uo pipefail` (no `-e`), which changes the
   whole `[ -x "$PUSH" ] && cmd` analysis and makes `${VAR:-default}`
   `set -u`-safety the live question instead. State it per file with line
   numbers (item 4's generic "set -e/-u interaction" is too weak on its own).
8. **Structural-scanner tests (regex + hardcoded path tuple) get named
   FN/FP axes.** For a scanner like `_EXEC_SITE` +
   `WRAPPERS: tuple[str, ...]`: a new wrapper absent from the tuple; a call
   shape whose arg is not the quoted string the regex demands; **`search()`
   binds only the FIRST match per line and the site counter counts LINES,
   not matches**; and the FP direction (can it match a guard/def line and
   fleet-block a healthy tree?). Also have the twin verify the docstring's
   own non-match claims (`[ -x "$PUSH" ]`, `"${PUSH_TIMEOUT}s"`) against the
   real wrapper text rather than trusting the prose.
9. **Split the pre-fix-failure duty BY TEST TYPE with a `not-shown` option.**
   A structural scanner IS statically verifiable — hand-apply its regex to
   `git show <sha>~1:<wrapper>` and to HEAD. Sleeping-stub TIMING tests are
   not: the implementer legitimately skips their pre-fix demo because each
   would hang to a designed `TimeoutExpired`. Compose a per-test
   `**Pre-fix failure evidence:** T-A <holds|not-shown|refuted> / ...` header
   line so the twin STATES the gap instead of fabricating a demo or silently
   ignoring the weakest pin.
10. **Manifest-grain pin-sweep residue is a known Step 4.6 shape.** Editing
    `tests/step9c_workflow_invariant_manifest.txt` makes that filename a
    changed literal whose grain surfaces invariant-pin test files the
    marker's `--map-files` + grep supplement legitimately miss (#2387: 3
    files). Pre-verify their `WORKFLOW_INVARIANT` membership (tuple AND
    manifest) so the routing is stated up front as Minor-at-most bookkeeping
    the Step 9c gate covers — never a `marker-shape` blocker.
11. **A `--map-files`-based plan verification step can be structurally
    self-defeating.** #2387's plan §4.5 asked for a `--map-files` probe
    showing the new test with a `literal-path:` reason, but registering that
    test in `WORKFLOW_INVARIANT` excludes it from `--map-files` output by
    design — so the plan's own probe goes empty the moment the plan's own
    registration step lands. Compose the disclosed substitution as a named
    `**Plan-deviation adjudication (§X):** upheld | rejected` line, with the
    duty to verify the structural claim FROM THE SELECTOR SOURCE (true +
    disclosed + equivalent substitute = plan imprecision; false claim or a
    weaker substitute = substantive).

**#2387 r2 (2026-08-29) — SCANNER-HARDENING closure round (the round-2 diff
tightens the r1 structural pin itself; one test file, no wrapper touched):**

12. **PASS+CONCERNS 5c-ter bounce = the twin's OWN concerns, no reconciler.**
    Both r1 verdicts were PASS-class (Claude PASS, Codex CONCERNS), so no
    reconciler was owed; the ledger rows are the twin's own `CONCERN::`
    output and Step 5c-ter blocked advance until addressed. Inline the
    orchestrator's `[round-boundary decision]` `epm:progress` note VERBATIM as
    the BOUNCE CONTRACT envelope — it carries the per-concern prescribed
    changes AND (load-bearing) the explicit "do not change control flow to
    close it" instruction. Author-neutrality binds both directions. Related:
    [[concern-discharge-round-severity-fence]] (#2552 r5 own-CONCERNS shape).
13. **THIRD fence type: a concern closed by REPORT CORRECTION ONLY, where the
    plan BARS the code fix.** The two CONCERNs took different fences in one
    ledger: the code-fix item got the upheld-bounce bar (NOT-ADDRESSED =
    substantive FAIL, PARTIALLY-ADDRESSED carve-out), while the
    report-correction item's deliverable is *the text being TRUE* — so compose
    (a) an explicit OUT-OF-BOUNDS line ("demanding a control-flow change
    cannot ground a finding"; the plan's must-ask list bars it), (b) a
    materially-FALSE-attribution route at the ordinary bar (misreporting), and
    (c) an honest-but-imprecise route at CONCERN via a same-id row. Without
    (a) the twin predictably re-raises the behavior it already ruled
    plan-sanctioned at r1.
14. **The highest-value composer table is the DUAL-PREDICATE mutant matrix —
    and its FP half is the part the round never self-checks.** Hand-apply BOTH
    the r1 and r2 predicates to CONSTRUCTED lines (no file mutated, nothing
    executed; a text-matching pin is fully statically decidable). That
    reproduced the marker's 3-mutant claim AND surfaced four shapes the marker
    never probed: an inline env-default (`"${PUSH_TIMEOUT:-20}s"`), a changed
    grace (`--kill-after=10s`), an unquoted duration, and a line-continuation
    split — each now REJECTED by the hardened pin. Hand them severity-unresolved
    with the question named ("correct pinning, acceptable friction, or
    over-tight?"); an over-tight pin on a `WORKFLOW_INVARIANT` member
    fleet-blocks every Step 9c run, so the FP direction weighs equal to the FN
    one. **Also check the loosening axis:** the r2 regex used `\s+` where the
    r1 test used a literal-space substring, so it is MORE permissive on
    whitespace than both r1 and the twin's own prescribed literal Fix — a
    hardening round is least likely to self-check the one axis it relaxed.
15. **Verify the report CORRECTION's citations yourself, hand over the
    judgment.** For a `(d)` retraction, read every cited `file:line` (all six
    arms verified exact here) and any magnitude claim (a read-only `crontab -l`
    settled "0 live watch crons" + the three live entries' exact times).
    Attest those as SETTLED, then hand the twin only what reading cannot
    settle: is the correction COMPLETE, does any r1 sentence still stand
    wrong, is "permanent" right given the re-arm path, and is a report-only
    close ADEQUATE (route that answer to BOTH the body and the ledger line —
    the orchestrator machine-acts on the ledger line).
16. **Composer self-observation trap, head-sentinel variant (#2326 r4 again):**
    writing "the body carries a versioned head sentinel `<!-- epm:results v2
    -->`" puts that literal in YOUR prose. Assert marker-side ==1 AND
    template-side ==1 AND total ==2 — never a bare total, which hides either
    side going to zero. Same split for a pinned SHA range: the inlined prior
    verdict legitimately carries the PRIOR round's `sha-range` line, so scope
    the staleness assert to the template.
17. **`grep -c` on a compose-time probe counts LINES;** the round-new-symbol
    sweep is cleanest stated as three path grains (full / basename / bare
    token) agreeing on one hit set, with generic-token grains
    (`n_sites`, 17 files) named do-not-promote up front.
