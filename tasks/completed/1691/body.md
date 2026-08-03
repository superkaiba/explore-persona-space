---
title: 'daily-fix: escalate multi-day Codex quota outage'
kind: infra
tags:
- wf-fix
- wf-fix-fp:feeb40f76ead
- daily-auto-filed
created_at: '2026-07-26T06:59:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): The Codex quota sentinel
  has been live since 2026-07-08 and runs to 2026-08-06, so every doubled review site
  across at least 21 sessions on 2026-07-25 degraded to single-Claude with no human-visible
  alarm and no reconciler invocation possible.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 problem sweep. Four independent transcript
miners, covering 21 sessions, each surfaced this without prompting: the Codex twin
half of the ensemble review architecture has been OFF fleet-wide for 17 days and
nothing surfaces that to a human.

## Goal

Add a watcher pass that pushes ONE deduped alert per Codex quota-outage episode when
the sentinel has been continuously live beyond a threshold, naming the outage window
and the count of review rounds that ran single-Claude since it opened.

## Workflow gap

- **Bug observed:** `.claude/cache/codex-quota-exhausted-until` has been live since
  **2026-07-08T08:23:26Z** and runs to **2026-08-06T13:26:00Z** — a ~29-day window,
  17 days elapsed, ~12 remaining. The #1204 pre-spawn check correctly short-circuits
  every `codex-*` composer, each session logs one `epm:progress` note, and the round
  proceeds single-Claude under the documented no-show fallback. That is exactly the
  designed fail-safe behaviour — the gap is that **nothing aggregates it**. A
  multi-week loss of the project's explicitly-named "strongest oversight asset"
  (CLAUDE.md § Codex ensemble review: cross-family reviewer diversity) is invisible
  outside the individual session logs.
- **Measured blast radius on 2026-07-25 alone:** ≥25 Codex twin dispatches skipped
  across the #1667–#1689 wave. Per-session `CODEX_QUOTA_LIVE` firing events (counted
  as `tool_result` blocks whose OUTPUT text contains the string, deduplicated per
  `tool_use_id`; `tool_use` command echoes and CLAUDE.md recipe text excluded): #1667=2,
  #1668=2, #1669=1, #1670=1, #1671=2, #1672=1, #1679=1, #1680=2, #1683=2, plus #1688
  and #1689. **Zero reconciler invocations occurred all day** — a direct consequence,
  since with no twin there can be no PASS-vs-FAIL disagreement to reconcile.
- **Why it is a workflow gap:** the sentinel is designed to fail silently-safe (the
  right call for a single round). There is no lane that notices "silently-safe" has
  been the steady state for two and a half weeks. Every other fleet-health signal in
  the project (disk pressure, stale pods, wedged sessions, registry drift,
  completed-unmerged) has a watcher pass with a deduped push; this one does not.
- **Confidence (emitter):** high on the outage and its scope; medium on the threshold
  (72 h vs 24 h vs 7 d is the planner's call).
- verified-at-filing: absence confirmed in the named target —
  `grep -c 'codex.quota\|codex-quota' scripts/autonomous_session_watch.py` → **0
  hits** (no pass reads the sentinel today). Sentinel state read directly at compose
  time from `.claude/cache/codex-quota-exhausted-until`:
  `{"until_unix": 1786022760.0, "until_iso": "2026-08-06T13:26:00+00:00",
  "parse_ok": true, "detected_at_iso": "2026-07-08T08:23:26+00:00",
  "job_id": "manual-seed-post-merge-1126"}`. Per-session skip counts read from the
  wave's transcripts by the sweep's miners under the firing-event counting rule
  (method stated above). (2026-07-25)

## Proposed change (refine in planning)

```
+ new pass in scripts/autonomous_session_watch.py, modelled on the existing
+ deduped-push passes (gate-push / registry-drift / completed-unmerged):
+   read .claude/cache/codex-quota-exhausted-until with the SAME two-sided
+   plausibility window the #1204 canonical check uses (now < until <= now + 45 d);
+   if detected_at is older than EPM_CODEX_OUTAGE_ALERT_HOURS (default 72),
+   push ONE alert per episode (episode key = detected_at_iso), naming:
+     - the window (detected_at -> until_iso) and days elapsed/remaining
+     - a count of review rounds that ran single-Claude since detected_at
+   sidecar row -> .claude/cache/codex-outage-events.jsonl
+   kill switch EPM_DISABLE_CODEX_OUTAGE_PASS=1; fail-open on unreadable sentinel
```

The single-Claude round COUNT is the part worth care: it needs a cheap source. Options
for the planner — count `epm:progress` notes matching the `codex composers skipped`
phrase across recent tasks, or have the #1204 skip-note path append a sidecar row and
count those. Prefer whichever avoids a fleet-wide events scan on every 10-min tick.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py` (a new pass).
- CLAUDE.md § Codex ensemble review — one sentence noting the escalation pass exists
  and naming its kill switch, so a future reader knows silence means "no outage".
- Do NOT change the #1204 pre-spawn check, the sentinel's lifecycle (still owned by
  `codex_task.py`), or the no-show fallback. This is an observer, exactly like the
  other escalate-only passes.

## Constraints / invariants

- Read-only w.r.t. the sentinel: the pass must never delete or rewrite it (lifecycle
  stays with the helper — the #1204 check is documented as read-only for this reason).
- One push per episode, TTL-deduped like the sibling passes; a 29-day outage must not
  produce 29 pushes.
- Fail-open: unreadable/corrupt/expired sentinel → no alert, no crash.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes;
  add a pin test alongside the existing watcher-pass tests.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Note on scope boundary

This task adds the ALARM only. Whether to upgrade, rotate, or accept the Codex
account state for the remaining ~12 days is Thomas's call and is tracked separately as
a `daily-held` `needs-human` task from the same sweep — do not attempt an account or
billing change from this session.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: feeb40f76ead
- Source: `/daily` 2026-07-25 transcript sweep — surfaced independently by four
  miners across sessions #1667–#1672, #1679/#1680/#1683, #1673–#1687, #1688/#1689.
