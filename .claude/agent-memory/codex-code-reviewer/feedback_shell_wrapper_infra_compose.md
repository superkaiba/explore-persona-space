---
name: shell-wrapper-infra-compose
description: Compose adaptations for kind:infra diffs touching a cron/shell wrapper (.sh) — Step 0.70 binds, live-alert + crontab-mutation never-run bans, seam=child-binary 3.8 shape, class-sweep completeness settle, scanner FN/FP axes, necessity-claim probes on concern-closure rounds
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

**#2386 r1 (2026-08-29) — SIBLING-SWEEP variant (the round ports the #2196
fail-loud log-dir guard to the ten sibling wrappers; 10 `cron_*.sh` + 1
selector row + 4 test files, +934/−25):**

18. **The class-boundary settle has a CLOSED ARITHMETIC form here — use it.**
    `git ls-tree -r --name-only origin/main -- scripts/ | grep -cE 'cron_.*\.sh$'`
    = 15, and FIX SET (10, all `M`) + NOT-APPLICABLE (5, untouched) = 15 with
    no residue. Stating the partition as exhaustive closes the
    false-incompleteness channel outright, far more cheaply than #2387's
    caller-by-caller classification. Still hand over the boundary question
    itself (a non-`cron_`-prefixed `.sh` with the same shape) — the ~170-file
    `mkdir -p` grep over `scripts/*.sh` is almost all per-issue dispatch
    scripts, so name that as the reason it is not in scope rather than
    leaving the twin to re-derive it.
19. **Verifying the NOT-APPLICABLE calls is the highest-value composer work,
    and one of them will usually have a pre-existing wart.** Check all five
    against the FILE, hand over evidence not conclusions. #2386's
    `cron_codex_auto_upgrade.sh` carries a residual comment claiming a
    checked `mkdir -p` "catches" exists-but-unwritable — which it does not
    (that is leg 2's entire premise). The file is UNTOUCHED by the round, so
    compose it as Step 0.9 `pre-existing-on-trunk` with an explicitly
    NARROWED reviewable question ("does the inaccuracy undermine the
    NOT-APPLICABLE call?") routed to CONCERN-at-most. Without the narrowing
    the twin predictably builds a round blocker on a comment from another
    task's commit.
20. **The both-legs probe MUST be multiline-aware.** The guard ships as
    `mkdir -p "$LOG_DIR" \` + `    || fatal "..."` on the next physical line,
    so a single-line `grep -c '|| fatal'` on the mkdir line returns 0 for all
    ten and reads as a total miss. Use `grep -A1 '^mkdir -p ' | grep -c '|| fatal'`.
    Same trap for the probe leg (`: >> "$LOG_FILE" 2>/dev/null \`).
21. **Attest the VACUOUS ordering case explicitly.** The probe must follow any
    first-run-of-day read (it CREATES the file and would flip the flag), but
    one wrapper (`cron_session_summarize`) has no such read at all, so the
    constraint is vacuous there and the file says so in a comment. State it,
    or the twin hunts for a missing `FIRST_RUN_OF_DAY` and reports its absence
    as the defect. Give the full per-wrapper table (fatal@ / mkdir@ / probe@ /
    FRoD@) and say plainly that line order is not control flow — the
    verify-against-real-control-flow duty is what the table sets up, not what
    it discharges.
22. **Test-harness literal coupling: check the driven set's PATTERN before
    flagging a substring mismatch.** `_MKDIR_FATAL = "cannot create log dir"`
    is NOT a substring of the Pattern B wrappers' `cannot create log/sentinel
    dir` — but the seven subprocess-driven wrappers are ALL Pattern A, and the
    two Pattern B ones have their own test files, so there is no present
    defect. Attest that partition; hand the latent-fragility judgment over.
23. **Pre-split (#1810) multi-unit builds need their own prompt block.** Units
    1-3 posted NO implementation marker by design and only unit 4's single
    `epm:results` covers the round. Say so explicitly, name the `unit N:`
    commit-subject prefixes, and state that earlier-unit marker absence is not
    a `marker-shape` defect — otherwise the twin reads 10 commits against 1
    marker as under-reporting.
24. **Composer-run gate-scope pre-read pays for itself.** #2386's
    `**Gate-scope check (#1288):**` field was complete (n_tests + base,
    locally-run list, dedup 14-file hit list, TWO `sweep_scope:` tokens,
    deferred count 137, NOT-RUN discharge with `recommended_timeout_s`), so
    the presence half discharges at compose time — state that and keep only
    the diff-consistency half, which is substantive and needs the twin's OWN
    `tests/` enumeration.

**#2387 r3 (2026-08-29) — NECESSITY-CLAIM variant (the round adds a
quote-aware bash-comment stripper + `\s+`→`[ \t]+`; one test file, +184/−19,
closing the twin's OWN r2 CONCERN):**

25. **When the implementer justifies a mechanism as NECESSARY, probe the
    necessity by running the SIMPLER alternative over the real tree.** #2387's
    `(a)` said quote-awareness was needed "by necessity, not by taste: every
    watch-script push line carries a `#` inside its double-quoted message, so a
    naive first-hash truncation would cut live sites short." Composer ran a
    quote-BLIND stripper beside the shipped one over all six wrappers: identical
    site sets and identical bound flags. Cause: `_EXEC_SITE` ends at the
    message's OPENING quote, and the `#` sits past the match end (`1739:132`
    match `[63,72)`, hash at 76). Only 2 of 10 site lines contain a `#` at all,
    so the universal quantifier fails too. Also find the shape where the
    mechanism IS load-bearing (here: a `#` inside an EARLIER quoted arg —
    quote-aware 1 site vs blind 0) so the twin gets both directions and can rate
    it defensive-but-mis-justified rather than useless. This is the highest-value
    composer probe on any round whose diff adds machinery to satisfy a concern.
26. **A necessity claim that fails usually drags its CONTROL TEST with it.**
    `test_live_push_line_scans_as_one_bounded_site` was docstring'd "the
    over-strip control (a naive comment strip would truncate the line here)" —
    it passes identically with quote tracking removed, so it cannot fail when
    the guarded mechanism is deleted. Hand that to the twin as a Step 4.5
    substance question over EVERY new test, not just the one you caught, and
    ask separately whether it bears on the CLOSURE at all (the strip mechanism
    still catches the concern's named mutant, so the answer may be "no").
27. **A disclosed residual's stated FAILURE DIRECTION is itself checkable, and
    can be backwards.** `(d)` disclosed the incomplete comment word-start set
    `" \t;&|("` and said "the failure direction is over-stripping, which drops
    a site and fails the count assertion loudly." Bash's word-start
    metacharacters also include `)`, `<`, `>`; composer probe shows a `#` after
    any of those is NOT stripped — UNDER-stripping, which retains a disabled
    push silently, the exact shape the open concern names. Give the table, name
    the realism question (is `pattern)# comment` a real cron-wrapper shape?),
    and SAY you did not run bash to confirm bash's own lexing — honest scope on
    a composer probe is what keeps it a fact rather than a verdict.
28. **Mid-compose ledger drift (#2326) has TWO row classes — keep the pin, but
    switch the row-emission rule to DISPOSITION-driven.** Between the ledger
    read and the build, two rows landed: the implementer's `addressed` row for
    the round's own concern, and the parallel Claude reviewer's fresh `raised`
    row (`raised_at_round == this round`). The `ts <= impl-marker-ts` pin
    correctly excludes both. But a snapshot-state-driven instruction ("the id is
    OPEN, so never emit a row") goes STALE: at verdict-forward time the latest
    event is `addressed`, and a partial closure then needs a same-id row to
    re-open. Write the rule as `VERIFIED-ADDRESSED ⇒ no row` /
    `PARTIALLY|NOT-ADDRESSED ⇒ same-id row`, and describe the snapshot as
    "pinned to the marker ts" rather than asserting a live ledger state. Report
    both excluded rows to the orchestrator; never inline the reviewer one.
29. **Author-neutrality needs the concern's OWN optionality quoted.** The r2
    Fix line marked its third item "For stronger protection, anchor the
    recognized execution shapes" — the round DECLINED it in `(b)`. Quote that
    phrasing into the neutrality block, or the twin re-reads its own Fix as
    three mandatory items and FAILs its own round-3 fix for the item it had
    itself marked secondary.
