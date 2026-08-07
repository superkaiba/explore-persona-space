---
title: 'dispatch_issue.py launch on the fellows lane: 600s SIGTERM disarms the QoS
  ladder; plan-copied commands drift (--repo-branch defaults to main)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-07T01:29:33Z'
has_clean_result: false
origin_prompt: 'Filed by the autonomous /issue 1336 session 2026-08-07 from an epm:failure-lesson
  v1 emitted by the experimenter during round 4''s production launch (SLURM 4684,
  fellows lane). Two confirmed gaps: the launch path blocks past the 600s Bash cap
  so a SIGTERM disarms the #1899 QoS fallback ladder after a successful submit; and
  plan-copied dispatch commands drift from the CLI, including --repo-branch silently
  defaulting to main on the SLURM rsync lane (would run unreviewed code) and --max-run-duration
  being GCP-only-inert against a 4h SLURM default.'
workflow: v1
---
# `dispatch_issue.py launch` on the fellows lane: 600s SIGTERM disarms the QoS ladder, and plan-copied commands silently drift

<!-- workflow-fix-candidate v1 -->

Filed by the autonomous `/issue 1336` session (2026-08-07) from an `epm:failure-lesson v1` the `experimenter` emitted while launching round 4's production run (SLURM job 4684, fellows lane, 8x H200). `root_cause_confirmed: yes`, `generalizes: yes`. The launch SUCCEEDED, so nothing is broken on #1336 — but two workflow-surface gaps are now demonstrated and both can silently damage a future run.

## Gap 1 — the 600s Bash cap SIGTERMs a launch that already succeeded, and the kill disarms the QoS fallback ladder

**Target:** `scripts/dispatch_issue.py` (launch path), `.claude/skills/issue/SKILL.md` Step 6d, `.claude/rules/compute-backend-failover.md`.

On the fellows lane, `dispatch_issue.py launch` does NOT return after `sbatch`. It blocks in the router-side queue-park wait. The Bash tool's 600 s cap therefore SIGTERMs it — AFTER submit and after the handle sidecar is written. Two consequences:

1. **The launch looks failed but is not.** The job is queued and the sidecar exists. Any naive retry double-submits. On #1336 the experimenter caught this and reported the correct recovery (`scancel <job>`, confirm dead, THEN re-run — never re-run while the job is queued), but nothing in the workflow surface states the invariant, so the next agent has to rediscover it.
2. **The #1899 QoS park-timeout fallback ladder is silently DISARMED.** That ladder (`high-eur` -> `normal-eur` -> `low-eur` -> RunPod) lives router-side in the process that just got killed. So a job that parks indefinitely in `high-eur` will never walk down to a lower QoS or fall back to RunPod — the exact protection the ladder exists to provide is gone, with no marker and no warning. On #1336 the job is PENDING (Resources) with a worst-case backfill estimate ~18 h out, and the orchestrator now has to own the ladder by hand.

**Fix candidates.** (a) Make the launch path return immediately after submit + sidecar write, and move the queue-park wait into a resumable orchestrator-side poll (the same shape the pod lane already uses). (b) Failing that, arm the QoS ladder as durable state — a sidecar field the poller re-reads — so it survives the launcher's death. (c) At minimum, document the probe-then-rearm recipe in SKILL.md Step 6d: probe sidecar + `squeue` BEFORE any relaunch, and re-drive only via scancel-then-re-run.

## Gap 2 — plan-copied dispatch commands drift from the real CLI, including one silent wrong-code hazard

**Target:** `.claude/agents/planner.md` §9 (launch-command composition), `scripts/dispatch_issue.py` (defaults), `.claude/skills/issue/SKILL.md` Step 6d.

Plan v15 §9 carried a launch command that three ways did not match the CLI. The experimenter caught all three and verified each against the code, but a less careful agent would not have:

1. **`launch` subcommand is argparse-required** — the plan's command omitted it and would simply have errored. Loud, harmless.
2. **`--repo-branch` defaults to `main` on the SLURM rsync lane.** This is the dangerous one. A plan's copy-pasted command runs **main's code, not the reviewed feature-branch code**. On #1336 it failed loud only by luck — `main` has no `all_v3` phase, so it would have errored. On any task whose phase name also exists on main, the lane would have silently run UNREVIEWED code while every marker claimed the reviewed SHA. That is a correctness hazard, not an ergonomics one.
3. **`--max-run-duration` is GCP-only and inert on SLURM**, while the SLURM `capture-7b` default `--time` is 4 h. Round 4's projected wall is ~8 h, so the plan's intended 24 h fence would have evaporated and the job would have been **killed at 4 h mid-run**. The fence has to be threaded as `--time-budget-hours`.

**Fix candidates.** (a) Have the planner emit lane-correct launch commands, or emit an intent spec that the dispatcher renders, rather than a hand-written command string that rots. (b) Make `--repo-branch` REQUIRED (or default to the issue's worktree branch) on any lane that rsyncs a repo — silently defaulting to `main` for an issue-branch workload is the wrong default. (c) Reject or warn on GCP-only flags when the resolved lane is SLURM, instead of accepting them inertly. (d) Add a verify_plan check that a plan-embedded launch command parses against the real CLI.

## Evidence

- #1336 `epm:run-launched` (2026-08-07, note leads `pod=eps-issue-1336-superkaiba`) and the `epm:failure-lesson v1` block in the experimenter's return.
- SLURM job 4684 `eps-issue-1336-superkaiba` on charmander, partition general, QOS high-eur, `cpu=64,mem=1T,gres/gpu=8`, TimeLimit 1-00:00:00 (realized from `--time-budget-hours 24`).
- Handle: `.claude/cache/issue-1336-handle.json` (backend=cluster/fellows).
- Command as actually executed, after the three corrections: `EPM_AUTO_LANE_ORDER=fellows,runpod uv run python scripts/dispatch_issue.py launch --issue 1336 --intent capture-7b --gpus 8 --repo-branch issue-1336-fullcorpora --time-budget-hours 24 --boot-disk-gb 500 --max-run-duration 24h --workload-cmd 'bash scripts/issue1336_dispatch.sh all_v3'`
- Plan v15 §9's original string, for the diff: same minus `launch`, minus `--repo-branch`, minus `--time-budget-hours`.

## Adjacent observation (not part of this filing's scope)

The same preflight run warned that HF public storage is at **16.05 TB against a 10 TB soft ceiling**, flagging LFS-403 risk for `upload_v3`'s tensor uploads. That is being tracked separately on #1336; it is noted here only because both surfaced from one preflight and a reader of this task may want the context.
