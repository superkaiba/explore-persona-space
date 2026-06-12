---
name: Infra-plan review checklist (guards, gates, tests, channels)
description: Umbrella for kind:infra plan reviews — protection-illusion holes, choke-point grep-verification, deviation-event observers, success-path-only tests, verification-matrix constructibility, channel-kill-vector writers, sentinel truthification (#564, #596, #607)
type: feedback
---

Recurring decision surface for `kind: infra` plan reviews. Infra plans live or die on integration-point correctness; the cheap independent verification is grep + reading the cited lines.

**A. Protection-illusion holes (#564, HF storage headroom)** — ways a guard looks armed but is inert:
1. *Partial-None probe blind spot:* a usage probe mapping per-item `None` → 0 (suspect-guard only on ALL-zero) understates when the ONE dominant item returns None (#541: 10.2 of 11.3 TB in one repo). Demand: count Nones, fold into `basis`, any-None-on-nonempty = suspect.
2. *Deviation events emitted pod-side need a named observer.* Pod code cannot post `epm:` markers; a "plan-deviation note" as pod-local log + JSONL dies at termination unless the filename matches the poller's sentinel glob (`/workspace/logs/issue-<N>-*.json`) or it uploads as non-LFS text. Grep poll_pipeline.py / upload-verifier.md for the event name — emission with zero wired observer is prose, not a channel. Usually a Concern (doc edit in scope).
3. *Opt-in flag interplay:* reroute keyed on a signal + a kill switch on the signal source means FLAG=1 + CHECK=0 is silently status quo — demand a loud warning on the armed-but-blind combo.
4. *Soft-ceiling vs hard-wall conflation:* a gate aborting in the deliberate runway zone must not claim "doomed" — old behavior succeeds there; wording + autonomous-blockage cost should be named.
These are usually implementer-absorbable Concerns; REVISE only if the mechanism would not fire in the very incident it was designed for, or a failure path is WORSE than status quo (prefer fail-open on unknown).

**B. Choke-point claims are grep-checkable — do it (#564).** When a plan claims "function X is THE funnel for behavior Y", grep the call sites and compare against the plan's enumeration; a missed caller is a silent routing hole, an exact match is strong APPROVE evidence.

**C. Test-plan holes (#607, GCP startup-script SIGPIPE):**
1. *Success-path-only integration tests:* a binding criterion "failure mode X can no longer happen, verified by test T" needs T to actually TRIGGER X's failure branch (trap fires, failed-phase published, shutdown stub invoked) — exit-0 + a string assert on the guard's presence measures nothing.
2. *Producer/parser contracts tested parser-only:* a producer→parser key contract over SSH stdout (e.g. `EPS_LOG_MTIME=`) tested by injecting keys into hand-crafted fixtures leaves a producer-side quoting/typo bug silently un-fired with all tests green; demand a producer-side command-string assert next to the parser test.
3. *Read-window arithmetic on negative asserts:* check the asserted pattern can even fit in the bytes read ("no 'x'*1000 run in ≤64 bytes" is vacuous).

**D. Verification-matrix constructibility (#596, sparse worktrees):** walk each §5 verification claim and ask: can the fixture as designed ever PRODUCE an instance of the claimed case? (#596 claimed out-of-cone-committed-path coverage but every fixture path became in-cone before commit.) Also disk-size methodology: `du` on a long-lived checkout conflates tracked bytes with untracked litter — measure on a FRESH worktree (#596: claimed 14G/worktree, fresh truth 3.8G + 11G .venv). Concerns when the unconstructible case has independent real-repo evidence + a one-line fixture fix, and corrected numbers still clear the criterion.

**E. Channel-kill-vector hardening (#607):** enumerate ALL residual writers to the hazardous channel and verify each is per-line bounded + error-guarded (`tail -n K | cut -c1-M` is safe; bare `tail -n K` of a file that can hold one giant newline-free line is NOT — `-n` counts lines). Check whether the consumer side already truncates (then a giant line is a transient blip → Concern). When a plan truthifies a hardwired placeholder metric, grep ALL consumers for sentinel-keyed semantics (`== 10**9` skip branches, vacuously-true predicates) — placeholder→truthful that makes a predicate strictly more accurate is safe; a consumer using the placeholder as "unknown, skip" could newly fire.

Also re-apply feedback_full_suite_green_needs_baseline.md to any "full pytest green" / repo-wide ruff row in infra plans.
