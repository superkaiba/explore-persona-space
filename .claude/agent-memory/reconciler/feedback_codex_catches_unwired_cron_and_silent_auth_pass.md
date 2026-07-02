---
name: Codex catches unwired-cron + silent-auth-pass in infra-janitor plans
description: critic/statistics — Claude APPROVEs a cron-wiring plan whose acceptance set never installs the cron + whose live probe passes on a failed gcloud list; verify install-is-an-artifact + the reaper's return-[]-on-nonzero-rc.
type: feedback
---

On `kind: infra` "wire <reaper> to a recurring cron" plans, Claude's
plan-critic APPROVEs while missing two conclusion-defeating gaps that
Codex catches; both are REVISE-grade and verifiable from the artifact.
(#639 r1, statistics lens — Claude APPROVE vs Codex REVISE; I sided with
Codex.)

**Why:** the Goal of a janitor-wiring task is "the janitor actually FIRES
on a schedule." A plan can pass every §6/§10 acceptance check and still
leave that false.

**How to apply — three checks before crediting a cron-wiring APPROVE:**

1. **Is crontab installation an acceptance ARTIFACT, or hand-applied?**
   If the plan routes the crontab edit to "applied by hand" / "user
   applies after review" (an outside-the-worktree state change, which is
   correct on the safety axis) AND the acceptance set is hand-fire-only
   (`bash scripts/cron_*.sh` once), then a merge lands green with the
   script written but never scheduled → the janitor NEVER fires. In an
   autonomous session there is no user to install it. Demand a post-apply
   verification artifact: `crontab -l` contains the exact line + the first
   scheduled tick produces the dated log + the first-run-of-day pointer.
   (#639 §4.3 + §7 R3 + Acceptance #5.)

2. **Can the "live probe PASS" happen on a FAILED real API call?** Read
   the reaper function: the EPS GCP/RunPod reapers `return []` on a
   non-zero `gcloud/list` rc (gcp.py:1958-1964 — `logger.error` +
   `return []`, never raise). If the smoke accepts "a JSON list (possibly
   empty `[]`), exit 0, a transient auth/stderr line is acceptable," then
   an expired-auth / wrong-config janitor passes the smoke AND every tick
   while reaping nothing — the credit-leak backstop is silently dead.
   Demand the smoke PROVE a successful real list (rc=0 on a direct
   `gcloud compute instances list --configuration=eps-gcp`, or a logged
   `list_rc=0` field), not just "empty list + exit 0." This is the same
   silent-fail family as the methodology-side "auth/list failure passes
   smoke."

3. **Does the unit fixture cover the CLI's sole conclusion-bearing
   branch?** A thin-CLI-over-library wrapper's only non-trivial logic is
   usually its exit-code map (e.g. any `delete-failed` record → `return
   2`). If §10.1 scripts only SUCCESSFUL deletes
   (`delete_results=[(0,..),(0,..)]`), the rc=2 path is untested. Demand a
   delete-failed (CLI rc=2) case. (Inherited library-level probe-failure
   tests like `..._probe_failure_never_reaps_and_never_crashes` cover the
   FUNCTION, not the new CLI's exit contract — don't count them as the
   CLI test.)

Claude's concerns on these plans tend to be cosmetic (invalid
`--delete=False` argparse, a `GcpConfig(**__dict__)` splat to collapse) —
real but non-blocking. Don't let a clean-looking acceptance table that
"maps every criterion to an artifact" substitute for asking whether the
artifacts prove the thing actually RUNS.

Orthogonal handled-risk to NOT re-flag: such plans are often built
against a not-yet-on-main function signature (#639's 6-arg issue-634
`audit_stale_gcp_vms` with `terminal_phase_max_age_seconds` + `reason`
keys). If the plan makes the branch base a load-bearing BLOCK gate (§9
assumption + §7 risk + a pre-write grep that BLOCKs on a miss), that is a
handled risk, not a finding — verify the gate exists, then move on.
